import os
import sys
import time
import argparse
import datetime
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import RandomSampler

from diffusers.models import AutoencoderKL
from models.diffusion import Diffusion
from models.loss import SupConLoss
from models.builder import build_model
from models.recognition import HTRNet

from data_loader.HWT_dataset import HWTDataset, ContentData, letters
from data_loader.builder import build_dataloader
from data_loader.data_sampler import AspectRatioBatchSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from mmcv.runner import LogBuffer

from utils.misc import read_config, init_random_seed, set_random_seed
from utils.logger import rename_file_with_creation_time, get_root_logger
from utils.dist_utils import get_world_size, flush
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lr_scheduler

warnings.filterwarnings("ignore")  # ignore warning
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    im = torchvision.transforms.ToPILImage()(grid)
    im.save(path)
    return im


@torch.no_grad()
def log_validation(model, step, device, vae=None):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []
    test_loader_iter = iter(test_loader)
    test_data = next(test_loader_iter)
    # prepare input
    data_type = torch.float32
    if accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    elif accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    images, style_ref, laplace_ref, content_ref, img_hw, aspect_ratio = test_data['img'].to(device, dtype=data_type), \
        test_data['style'].to(device, dtype=data_type), \
        test_data['laplace'].to(device, dtype=data_type), \
        test_data['content'].to(device, dtype=data_type), \
        test_data['img_hw'].to(device, dtype=data_type), \
        test_data['aspect_ratio'].to(device, dtype=data_type)
    load_content = ContentData()
    texts = ['(Why,', '"21Feb', 'everyone,']
    if vae is None:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, subfolder="vae").to(accelerator.device)
        vae.requires_grad_(False)
    torch.cuda.empty_cache()

    for text in texts:
        torch.cuda.empty_cache()
        text_ref = load_content.get_content(text)
        text_ref = text_ref.to(accelerator.device).repeat(style_ref.shape[0], 1, 1, 1)
        x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(
            accelerator.device)
        preds = diffusion.ddim_sample(model, vae, images.shape[0], x, style_ref, laplace_ref, text_ref, config=config)
        out_path = os.path.join(os.path.join(config.work_dir, "save_sample_dir"), f"Step-{step}-{text}.png")
        save_images(preds, out_path)

    del vae
    del images, style_ref, laplace_ref, content_ref, img_hw, aspect_ratio, preds, text_ref
    flush()
    return image_logs


def train():
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    global_step = start_step + 1

    os.makedirs(os.path.join(config.work_dir, "save_sample_dir"), exist_ok=True)

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        for step, data in enumerate(train_loader):
            flush()
            if step < skip_step:
                global_step += 1
                continue  # skip data in the resumed ckpt
            data_type = torch.float32
            if accelerator.mixed_precision == "fp16":
                data_type = torch.float16
            elif accelerator.mixed_precision == "bf16":
                data_type = torch.bfloat16
            images, style_ref, laplace_ref, content_ref, wid = data['img'].to(accelerator.device, dtype=data_type), \
                data['style'].to(accelerator.device, dtype=data_type), \
                data['laplace'].to(accelerator.device, dtype=data_type), \
                data['content'].to(accelerator.device, dtype=data_type), \
                data['wid'].to(accelerator.device, dtype=data_type)
            if config.load_vae_feat:
                images = data['img'].to(accelerator.device, dtype=data_type)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                            enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        images = vae.encode(images).latent_dist.sample() * config.scale_factor

            # forward
            t = diffusion.sample_timesteps(images.shape[0]).to(accelerator.device)
            x_t, noise = diffusion.noise_images(images, t)
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                predicted_noise, high_nce_emb, low_nce_emb = model(x_t, t, style_ref, laplace_ref, content_ref,
                                                                   tag='train')
                # calculate loss
                recon_loss = recon_criterion(predicted_noise, noise)
                high_nce_loss = nce_criterion(high_nce_emb, labels=wid)
                low_nce_loss = nce_criterion(low_nce_emb, labels=wid)
                loss = recon_loss + high_nce_loss + low_nce_loss

                accelerator.backward(loss)

                del predicted_noise, high_nce_emb, low_nce_emb
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            logs.update(recon_loss=accelerator.gather(recon_loss).mean().item())
            logs.update(high_nce_loss=accelerator.gather(high_nce_loss).mean().item())
            logs.update(low_nce_loss=accelerator.gather(low_nce_loss).mean().item())
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()

                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)
            global_step += 1
            data_time_start = time.time()

            if config.save_model_steps and global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=global_step,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )

            if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    log_validation(model, global_step, device=accelerator.device, vae=vae)
        if (epoch % config.save_model_epochs == 0 or epoch == config.num_epochs) and epoch > config.start_save_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )

        accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', default=None, help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Currently, we only support reporting to TensorBoard.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="HWDiT",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True
        )
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2
    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    fsdp_plugin = None
    even_batches = True
    if config.multi_scale:
        even_batches = False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )
    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))
    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")

    vae = None
    if not config.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, subfolder="vae").to(accelerator.device)
        vae.requires_grad_(False)
    """ MODEL """
    model_kwargs = {"qk_norm": config.qk_norm, "differential": config.differential}
    model = build_model(config.model,
                        config.grad_checkpointing,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    """load pretrained resnet18 model"""
    if config.feat_model != "":
        checkpoint = torch.load(config.feat_model, map_location=torch.device('cpu'))
        checkpoint['conv1.weight'] = checkpoint['conv1.weight'].mean(1).unsqueeze(1)
        miss, unexp = model.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)
        assert len(unexp) <= 32, "faile to load the pretrained model"
        print('load pretrained model from {}'.format(config.feat_model))

    if args.load_from is not None:
        config.load_from = args.load_from
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), resume_lr_scheduler=False,
            resume_optimizer=False)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    if config.load_ocr:
        '''load pretrained ocr model'''
        ocr_model = HTRNet(nclasses=len(letters), vae=True)
        if config.ocr_model is not None:
            miss, unxep = ocr_model.load_state_dict(torch.load(config.ocr_model, map_location=torch.device('cpu')),
                                                    strict=False)
            print('load pretrained ocr model from {}'.format(config.ocr_model))
        else:
            print('failed to load the pretrained ocr model')
            exit()
        ocr_model.requires_grad_(False)
        ocr_model = ocr_model.to(accelerator.device)

    """ set dataset """
    train_dataset = HWTDataset(config.IMAGE_PATH, config.STYLE_PATH, config.LAPLACE_PATH, "train",
                               load_vae_feat=config.load_vae_feat, vae_path=config.vae_path)
    print('number of training images: ', len(train_dataset))
    batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(train_dataset), dataset=train_dataset,
                                            batch_size=config.train_batch_size,
                                            aspect_ratios=train_dataset.aspect_ratio,
                                            drop_last=False,
                                            ratio_nums=train_dataset.ratio_nums, valid_num=config.valid_num)
    train_loader = build_dataloader(train_dataset, batch_sampler=batch_sampler, num_workers=config.num_workers,
                                    collate_fn=train_dataset.collate_fn_)
    test_dataset = HWTDataset(config.IMAGE_PATH, config.STYLE_PATH, config.LAPLACE_PATH, "test",
                              load_vae_feat=config.load_vae_feat, vae_path=config.vae_path)
    print('number of testing images: ', len(test_dataset))

    test_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(test_dataset), dataset=test_dataset,
                                                 batch_size=config.test_batch_size,
                                                 aspect_ratios=test_dataset.aspect_ratio,
                                                 drop_last=False,
                                                 ratio_nums=test_dataset.ratio_nums,
                                                 valid_num=config.valid_num)
    test_loader = build_dataloader(test_dataset, batch_sampler=test_batch_sampler, num_workers=config.num_workers,
                                   collate_fn=test_dataset.collate_fn_)
    """ build optimizer and lr scheduler """
    lr_scale_ratio = 1
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_loader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step
    total_steps = len(train_loader) * config.num_epochs

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    """ Prepare everything """
    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_loader, lr_scheduler)
    recon_criterion, nce_criterion, ctc_loss = nn.MSELoss(), SupConLoss(contrast_mode='all'), nn.CTCLoss()
    diffusion = Diffusion(device=accelerator.device, noise_offset=config.noise_offset)

    """build trainer"""
    train()
