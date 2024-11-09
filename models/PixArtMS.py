# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention
from timm.layers.mlp import SwiGLU
import torch.nn.functional as F
from einops import rearrange
from models.builder import MODELS
from models.utils import auto_grad_checkpoint, to_2tuple

from models.PixArt_blocks import t2i_modulate, RMSNorm, T2IFinalLayer,\
    TimestepEmbedder, CrossAttention, MultiheadDiffAttn, get_2d_sincos_pos_embed
from models.fusion import Mix_TR


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            patch_size=2,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PixArtMSBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., depth=12, context_dim=512, qk_norm=True, differential=True, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        
        if differential:
            self.norm1 = RMSNorm(hidden_size, eps=1e-5, elementwise_affine=True)
            self.norm2 = RMSNorm(hidden_size, eps=1e-5, elementwise_affine=True)
            self.attn = MultiheadDiffAttn(embed_dim=hidden_size,
                                        depth=depth, num_heads=num_heads, qk_norm=qk_norm)
            self.mlp = SwiGLU(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=0)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
            self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu,drop=0)

        self.cross_attn = CrossAttention(query_dim=hidden_size, context_dim=context_dim, heads=num_heads, dim_head=hidden_size//num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.cross_attn(x, y)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x

#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class PixArtMS(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=4,
                 mlp_ratio=4.0, drop_path: float = 0., context_dim=512, qk_norm=True, differential=True,**kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.out_channels = in_channels
        self.h = self.w = 0
        self.context_dim = context_dim
        self.differential = differential
        self.qk_norm = qk_norm
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.x_embedder = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtMSBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, drop_path=drop_path[i],
                          depth=self.depth, context_dim=self.context_dim, qk_norm=self.qk_norm, differential=self.differential)
            for i in range(self.depth)
        ])
        self.final_layer = T2IFinalLayer(self.hidden_size, self.patch_size, self.out_channels, self.differential)
        self.mix_net = Mix_TR(d_model=self.context_dim)
        self.initialize()

    def forward(self, x, timestep, style=None, laplace=None, content=None, tag='train', **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)

        if tag == 'train':
            context, high_nce_emb, low_nce_emb = self.mix_net(style, laplace, content)
        else:
            context = self.mix_net.generate(style, laplace, content)

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.hidden_size, (self.h, self.w))).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        t0 = self.t_block(t)
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, context, t0, **kwargs)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if tag == 'train':
            return x, high_nce_emb, low_nce_emb
        else:
            return x

    def forward_backbone(self, x, timestep, style=None, laplace=None, content=None, **kwargs):
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        context, high_nce_emb, low_nce_emb = self.mix_net(style.to(self.dtype), laplace.to(self.dtype),
                                                          content.to(self.dtype))
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.hidden_size, (self.h, self.w))).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        t0 = self.t_block(t)
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, context, t0, **kwargs)  # (N, T, D) #support grad checkpoint
        return x

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module()
def PixArtMS_XL_2(**kwargs):
    return PixArtMS(depth=18, hidden_size=1024, patch_size=2, num_heads=4, **kwargs)
