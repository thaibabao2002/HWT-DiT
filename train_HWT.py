import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers.models import AutoencoderKL
from torch.utils.data import RandomSampler, SequentialSampler

from models.unet import UNetModel
from models.diffusion import Diffusion, EMA
from data_loader.builder import build_dataset, build_dataloader, set_data_root
from data_loader.data_sampler import AspectRatioBatchSampler
from models.builder import build_model

from trainer.HWT_trainer import Trainer
from data_loader.HWT_dataset import HWTDataset
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from models.loss import SupConLoss
from utils.logger import set_log
from utils.util import fix_seed

warnings.filterwarnings("ignore")  # ignore warning

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set multi-gpu """
    # dist.init_process_group(backend='nccl')
    # local_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device(opt.device, local_rank)
    device = torch.device(opt.device)

    """ model """
    model_kwargs = {"lewei_scale": cfg.MODEL.lewei_scale, 'config': cfg,}

    model = build_model(cfg.MODEL.model,
                        cfg.MODEL.grad_checkpointing,
                        cfg.MODEL.get('fp32_attention', False),
                        pred_sigma=cfg.MODEL.pred_sigma,
                        **model_kwargs).train()

    """build model architecture"""
    # model = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM,
    #                  out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
    #                  attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS,
    #                  context_dim=cfg.MODEL.EMB_DIM).to(device)

    """load pretrained resnet18 model"""
    if len(opt.feat_model) > 0:
        checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'))
        checkpoint['conv1.weight'] = checkpoint['conv1.weight'].mean(1).unsqueeze(1)
        miss, unexp = model.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)
        assert len(unexp) <= 32, "faile to load the pretrained model"
        print('load pretrained model from {}'.format(opt.feat_model))

    if len(opt.one_dm) > 0:
        model.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))
    # model = DDP(model, device_ids=[local_rank])
    model = model.to(opt.device)

    """ set dataset"""
    train_dataset = HWTDataset(
        cfg.DATA_LOADER.IMAGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TRAIN.TYPE)
    print('number of training images: ', len(train_dataset))

    batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(train_dataset), dataset=train_dataset,
                                            batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                            aspect_ratios=train_dataset.aspect_ratio,
                                            drop_last=False,
                                            ratio_nums=train_dataset.ratio_nums, valid_num=10)
    train_loader = build_dataloader(train_dataset, batch_sampler=batch_sampler, num_workers=4,
                                    collate_fn=train_dataset.collate_fn_)
    # train_loader = build_dataloader(train_dataset, batch_size=128, num_workers=4,
    #                                 collate_fn=train_dataset.collate_fn_)

    test_dataset = HWTDataset(
        cfg.DATA_LOADER.IMAGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TEST.TYPE)
    print('number of testing images: ', len(test_dataset))

    test_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(test_dataset), dataset=test_dataset,
                                                 batch_size=cfg.TEST.IMS_PER_BATCH,
                                                 aspect_ratios=test_dataset.aspect_ratio,
                                                 drop_last=False,
                                                 ratio_nums=test_dataset.ratio_nums,
                                                 valid_num=10)
    test_loader = build_dataloader(test_dataset, batch_sampler=test_batch_sampler, num_workers=4,
                                   collate_fn=test_dataset.collate_fn_)
    # test_loader = build_dataloader(test_dataset, batch_size=64, num_workers=4,
    #                                collate_fn=test_dataset.collate_fn_)

    """build criterion and optimizer"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    """Freeze vae and text_encoder"""
    vae.requires_grad_(False)
    vae = vae.to(device)

    """build trainer"""
    trainer = Trainer(diffusion, model, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--feat_model', dest='feat_model', default='model_zoo/RN18_class_10400.pth', help='pre-trained resnet18 model')
    parser.add_argument('--one_dm', dest='one_dm', default='', help='pre-trained one_dm model')
    parser.add_argument('--log', default='English',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)
