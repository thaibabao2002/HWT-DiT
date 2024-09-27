import numpy as np
import torch
import torch.nn as nn
import os
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from models.fusion import Mix_TR
from models.sora.checkpoint import auto_grad_checkpoint
from models.sora.blocks import (
    Attention,
    MultiHeadCrossAttention,
    PatchEmbed2D,
    PositionEmbedding2D,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from models.sora.registry import MODELS
from transformers import PretrainedConfig, PreTrainedModel


class STDiT2Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            drop_path=0.0,
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False,
            qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self._enable_sequence_parallelism = enable_sequence_parallelism

        # spatial branch
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
            qk_norm=qk_norm,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

        # cross attn
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        # mlp branch
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y, t):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        # modulate
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = self.attn(x_m)
        x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # cross attn
        x = x + self.cross_attn(x, y)

        # modulate
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)

        # mlp
        x_mlp = self.mlp(x_m)
        x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


class STDiT2Config(PretrainedConfig):
    model_type = "STDiT2"

    def __init__(
            self,
            input_size=(None, None, None),
            input_sq_size=32,
            in_channels=4,
            patch_size=(2, 2),
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            pred_sigma=True,
            drop_path=0.0,
            caption_channels=4096,
            model_max_length=120,
            freeze=None,
            qk_norm=False,
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
            **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.freeze = freeze
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        super().__init__(**kwargs)


@MODELS.register_module()
class STDiT2(PreTrainedModel):
    config_class = STDiT2Config

    def __init__(
            self,
            config
    ):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)

        self.x_embedder = PatchEmbed2D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(),
                                          nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True))  # new
        self.mix_net = Mix_TR(d_model=self.hidden_size)
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        self.blocks = nn.ModuleList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    qk_norm=config.qk_norm,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if config.freeze is not None:
            assert config.freeze in ["not_temporal", "text"]
            if config.freeze == "not_temporal":
                self.freeze_not_temporal()
            elif config.freeze == "text":
                self.freeze_text()

    def get_dynamic_size(self, x):
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            H += self.patch_size[0] - H % self.patch_size[0]
        if W % self.patch_size[1] != 0:
            W += self.patch_size[1] - W % self.patch_size[1]
        H = H // self.patch_size[0]
        W = W // self.patch_size[1]
        return (H, W)

    def forward(
            self, x, timestep, style=None, laplace=None, content=None, height=None, width=None, ar=None, tag='train'
    ):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of image; of shape [B, C, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        B = x.shape[0]
        dtype = self.x_embedder.proj.weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)

        if tag=='train':
            context, high_nce_emb, low_nce_emb = self.mix_net(style, laplace, content)
        else:
            context = self.mix_net.generate(style, laplace, content)

        # === process data_loader info ===
        # 1. get dynamic size
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        rs = (height[0].item() * width[0].item()) ** 0.5
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = torch.cat([csize, ar], dim=1)

        # === get dynamic shape size ===
        _, _, Hx, Wx = x.size()
        H, W = self.get_dynamic_size(x)
        S = H * W
        scale = rs / self.input_sq_size
        base_size = round(S ** 0.5)
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = x + pos_emb

        # prepare adaIN
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_spc = t + data_info  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]

        # blocks
        for _, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block,
                x,
                context,
                t_spc_mlp,
            )
            # x.shape: [B, N, C]

        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, H, W, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        if tag == 'train':
            return x, high_nce_emb, low_nce_emb
        else:
            return x

    def unpatchify(self, x, N_h, N_w, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, H, W]
        """

        H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_h N_w) (H_p W_p C_out) -> B C_out (N_h H_p) (N_w W_p)",
            N_h=N_h,
            N_w=N_w,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_h, :R_w]
        return x

    def get_spatial_pos_embed(self, H, W, scale=1.0, base_size=None):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (H, W),
            scale=scale,
            base_size=base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def initialize_weights(self):
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
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@MODELS.register_module("STDiT2-XL/2")
def STDiT2_XL_2(**kwargs):
    # create a new model
    config = STDiT2Config(
        depth=24,
        hidden_size=768,
        patch_size=(2, 2),
        num_heads=16, **kwargs
    )
    model = STDiT2(config)
    return model
