import os
import sys
import time
import argparse
import datetime
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import RandomSampler

from diffusers.models import AutoencoderKL
from models.diffusion import Diffusion
from models.loss import SupConLoss
from models.builder import build_model

from data_loader.HWT_dataset import HWTDataset, ContentData
from data_loader.builder import build_dataloader
from data_loader.data_sampler import AspectRatioBatchSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast
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
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, subfolder="vae", torch_dtype=torch.float16).to(
            accelerator.device)
        vae.requires_grad_(False)
    torch.cuda.empty_cache()

    for text in texts:
        torch.cuda.empty_cache()
        text_ref = load_content.get_content(text)
        text_ref = text_ref.to(accelerator.device).repeat(style_ref.shape[0], 1, 1, 1)
        x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(
            accelerator.device)
        preds = diffusion.ddim_sample(model, vae, images.shape[0], x, style_ref, laplace_ref, text_ref, config=config, sampling_timesteps=4)
        out_path = os.path.join(os.path.join(config.work_dir, "save_sample_dir"), f"Step-{step}-{text}.png")
        save_images(preds, out_path)

    del vae
    del images, style_ref, laplace_ref, content_ref, img_hw, aspect_ratio, preds, text_ref
    flush()
    return image_logs

def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


def compute_distribution_matching_loss(latents, style_ref, laplace_ref, content_ref):
    original_latents = latents
    batch_size = latents.shape[0]
    with torch.no_grad():
        timesteps = torch.randint(
            min_step,
            min(max_step + 1, denoising_timestep),
            [batch_size],
            device=latents.device,
            dtype=torch.long
        )
        noisy_latents, noise = diffusion.noise_images(latents, timesteps)
        pred_fake_noise, _, _ = model_fake(noisy_latents, timesteps, style_ref, laplace_ref, content_ref, tag='train')
        alpha_hat_t = diffusion.alpha_hat[timesteps][:, None, None, None]
        pred_fake_image = (noisy_latents - (1 - alpha_hat_t).sqrt() * pred_fake_noise) / (
            alpha_hat_t.sqrt())

        pred_real_noise, _, _ = model_real(noisy_latents, timesteps, style_ref, laplace_ref, content_ref, tag='train')
        pred_real_image = (noisy_latents - (1 - alpha_hat_t).sqrt() * pred_real_noise) / (
            alpha_hat_t.sqrt())
        p_real = (latents - pred_real_image)
        p_fake = (latents - pred_fake_image)
        grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
        grad = torch.nan_to_num(grad)

    loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents - grad).detach().float(), reduction="mean")
    return loss

def compute_cls_logits(image, style_ref, laplace_ref, content_ref):
    timesteps = torch.randint(
        0, denoising_timestep, [image.shape[0]], device=image.device, dtype=torch.long
    )
    image, noise = diffusion.noise_images(image, timesteps)
    rep = model_fake.forward_backbone(
        image, timesteps, style_ref, laplace_ref, content_ref
    )
    logits = cls_pred_branch(rep[:, 0].to(next(cls_pred_branch.parameters()).dtype))
    return logits

def compute_generator_clean_cls_loss(generated_image, style_ref, laplace_ref, content_ref):
    pred_realism_on_fake_with_grad = compute_cls_logits(
        generated_image, style_ref, laplace_ref, content_ref
    )
    loss = F.softplus(-pred_realism_on_fake_with_grad).mean()
    return loss

def compute_guidance_clean_cls_loss(real_image, fake_image, style_ref, laplace_ref, content_ref):
    pred_realism_on_real = compute_cls_logits(
        real_image.detach(), style_ref, laplace_ref, content_ref
    )
    pred_realism_on_fake = compute_cls_logits(
        fake_image.detach(), style_ref, laplace_ref, content_ref
    )
    classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
    return classification_loss

def compute_loss_fake(latents, style_ref, laplace_ref, content_ref):
    latents = latents.detach()
    batch_size = latents.shape[0]
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        denoising_timestep,
        [batch_size],
        device=latents.device,
        dtype=torch.long
    )
    noisy_latents, noise = diffusion.noise_images(latents, timesteps)
    fake_noise_pred, _, _ = model_fake(noisy_latents, timesteps, style_ref, laplace_ref, content_ref, tag='train')
    fake_noise_pred = fake_noise_pred.float()

    # epsilon prediction loss
    loss_fake = torch.mean(
        (fake_noise_pred.float() - noise.float()) ** 2
    )
    return loss_fake

@torch.no_grad()
def sample_backward(noisy_image, style_ref, laplace_ref, content_ref):
    batch_size = noisy_image.shape[0]
    device = noisy_image.device

    selected_step = torch.randint(low=0, high=num_denoising_step, size=(1,), device=device, dtype=torch.long)
    selected_step = broadcast(selected_step, from_process=0)
    generated_image = noisy_image
    for constant in denoising_step_list[:selected_step]:
        current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long) * constant
        generated_noise, _, _ = model(noisy_image, current_timesteps, style_ref, laplace_ref, content_ref, tag='train')
        alpha_hat_t = diffusion.alpha_hat[current_timesteps][:, None, None, None]
        generated_image = (noisy_image - (1 - alpha_hat_t).sqrt() * generated_noise) / (
            alpha_hat_t.sqrt())

        next_timestep = current_timesteps - timestep_interval
        noisy_image, noise = diffusion.noise_images(generated_image, next_timestep)

    return_timesteps = denoising_step_list[selected_step] * torch.ones(batch_size, device=device, dtype=torch.long)
    return generated_image, return_timesteps


@torch.no_grad()
def prepare_denoising_data(noise, style_ref, laplace_ref, content_ref):
    clean_images, timesteps = sample_backward(torch.randn_like(noise), style_ref, laplace_ref, content_ref)
    noisy_image, noise = diffusion.noise_images(clean_images, timesteps)
    pure_noise_mask = (timesteps == (denoising_timestep - 1))
    noisy_image[pure_noise_mask] = noise[pure_noise_mask]

    return timesteps, noisy_image


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
            images, style_ref, laplace_ref, content_ref, wid, target, target_lengths = data['img'].to(accelerator.device, dtype=data_type), \
                data['style'].to(accelerator.device, dtype=data_type), \
                data['laplace'].to(accelerator.device, dtype=data_type), \
                data['content'].to(accelerator.device, dtype=data_type), \
                data['wid'].to(accelerator.device, dtype=data_type), \
                data['target'].to(accelerator.device, dtype=data_type), \
                data['target_lengths'].to(accelerator.device, dtype=torch.int16)
            if config.load_vae_feat:
                images = data['img'].to(accelerator.device, dtype=data_type)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                            enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        images = vae.encode(images).latent_dist.sample() * config.scale_factor

            # forward
            with accelerator.accumulate(model) and accelerator.accumulate(model_fake):
                noise = torch.randn_like(images, device=accelerator.device, dtype=data_type)
                timesteps, noisy_image = prepare_denoising_data(noise, style_ref, laplace_ref, content_ref)
                generated_noise, high_nce_emb, low_nce_emb  = model(
                    noisy_image, timesteps.long(), style_ref, laplace_ref, content_ref, tag='train')

                alpha_hat_t = diffusion.alpha_hat[timesteps][:, None, None, None]
                generated_image = (noisy_image - (1 - alpha_hat_t).sqrt() * generated_noise) / (alpha_hat_t.sqrt())
                if step % 10 == 0:
                    model_fake.requires_grad_(False)
                    loss_dmd = compute_distribution_matching_loss(generated_image, style_ref, laplace_ref, content_ref)
                    loss_cls = compute_generator_clean_cls_loss(generated_image, style_ref, laplace_ref, content_ref)
                    model_fake.requires_grad_(True)
                    for param in model_fake.mix_net.parameters():
                        param.requires_grad = False
                    generator_loss = loss_dmd + loss_cls * gen_cls_loss_weight
                    accelerator.backward(generator_loss)
                    generator_grad_norm = accelerator.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_fake.zero_grad()
                lr_scheduler.step()

                ### Guidance
                loss_fake_mean = compute_loss_fake(generated_image, style_ref, laplace_ref, content_ref)
                guidance_cls_loss = compute_guidance_clean_cls_loss(images, generated_image, style_ref, laplace_ref, content_ref)
                guidance_loss = loss_fake_mean + guidance_cls_loss * guidance_cls_loss_weight
                accelerator.backward(guidance_loss)
                guidance_grad_norm = accelerator.clip_grad_norm_(model_fake.parameters(), 10)
                optimizer_fake.step()
                optimizer_fake.zero_grad()
                optimizer.zero_grad()
                lr_scheduler_fake.step()
                data_time_all += time.time() - data_time_start

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(generator_loss).mean().item()}
            if step % 10 == 0:
                logs.update(loss_dmd=accelerator.gather(loss_dmd).mean().item())
                logs.update(loss_cls=accelerator.gather(loss_cls).mean().item())
                logs.update(grad_norm=accelerator.gather(generator_grad_norm).mean().item())
            logs.update(guidance_loss=accelerator.gather(guidance_loss).mean().item())
            logs.update(loss_fake_mean=accelerator.gather(loss_fake_mean).mean().item())
            logs.update(guidance_cls_loss=accelerator.gather(guidance_cls_loss).mean().item())
            logs.update(guidance_grad_norm=accelerator.gather(guidance_grad_norm).mean().item())

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
                    state_dict = dict(state_dict=model_fake.state_dict())
                    file_path = os.path.join(config.work_dir, 'checkpoints', f"fake_model_epoch_{epoch}.pth")
                    file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
                    torch.save(state_dict, file_path)

            if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # pass
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
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, subfolder="vae", torch_dtype=torch.bfloat16).to(
            accelerator.device)
        vae.requires_grad_(False)
    """ MODEL """
    #####
    model_kwargs = {"qk_norm": config.qk_norm, "differential": config.differential}
    model = build_model(config.model,
                        config.grad_checkpointing,
                        **model_kwargs).train()
    model_real = build_model(config.model,
                        config.grad_checkpointing,
                        **model_kwargs).train()
    model_fake = build_model(config.model,
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
            config.load_from, model, load_ema=config.get('load_ema', False), resume_optimizer=False,
            resume_lr_scheduler=False)
        missing_real, unexpected_real = load_checkpoint(
            config.load_from, model_real, load_ema=config.get('load_ema', False), resume_optimizer=False,
            resume_lr_scheduler=False)
        missing_fake, unexpected_fake = load_checkpoint(
            config.load_from, model_fake, load_ema=config.get('load_ema', False), resume_optimizer=False,
            resume_lr_scheduler=False)

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    """ set dataset """
    #####
    train_dataset = HWTDataset(config.IMAGE_PATH, config.STYLE_PATH, config.LAPLACE_PATH, "train",
                               load_vae_feat=config.load_vae_feat, vae_path=config.vae_path)
    print('number of training images: ', len(train_dataset))

    #####
    batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(train_dataset), dataset=train_dataset,
                                            batch_size=config.train_batch_size,
                                            aspect_ratios=train_dataset.aspect_ratio,
                                            drop_last=False,
                                            ratio_nums=train_dataset.ratio_nums, valid_num=config.valid_num)
    train_loader = build_dataloader(train_dataset, batch_sampler=batch_sampler, num_workers=4,
                                    collate_fn=train_dataset.collate_fn_)
    test_dataset = HWTDataset(config.IMAGE_PATH, config.STYLE_PATH, config.LAPLACE_PATH, "test",
                              load_vae_feat=config.load_vae_feat, vae_path=config.vae_path)
    print('number of testing images: ', len(test_dataset))

    #####
    test_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(test_dataset), dataset=test_dataset,
                                                 batch_size=config.test_batch_size,
                                                 aspect_ratios=test_dataset.aspect_ratio,
                                                 drop_last=False,
                                                 ratio_nums=test_dataset.ratio_nums,
                                                 valid_num=config.valid_num)
    test_loader = build_dataloader(test_dataset, batch_sampler=test_batch_sampler, num_workers=4,
                                   collate_fn=test_dataset.collate_fn_)

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

    """ build optimizer and lr scheduler """
    lr_scale_ratio = 1
    optimizer = build_optimizer(model, config.optimizer)
    optimizer_fake = build_optimizer(model_fake, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_loader, lr_scale_ratio)
    lr_scheduler_fake = build_lr_scheduler(config, optimizer_fake, train_loader, lr_scale_ratio)

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 resume_optimizer=True,
                                                 resume_lr_scheduler=True
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    """ Prepare everything """
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model_real.requires_grad_(False)
    model_real.to(accelerator.device, dtype=weight_dtype)

    model = accelerator.prepare(model)
    model_fake = accelerator.prepare(model_fake)
    for param in model.mix_net.parameters():
        param.requires_grad = False
    for param in model_fake.mix_net.parameters():
        param.requires_grad = False

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_loader, lr_scheduler)
    optimizer_fake, lr_scheduler_fake = accelerator.prepare(optimizer_fake, lr_scheduler_fake)
    recon_criterion, nce_criterion, ctc_criterion = nn.MSELoss(), SupConLoss(contrast_mode='all'), nn.CTCLoss()
    diffusion = Diffusion(device=accelerator.device, noise_offset=config.noise_offset, noise_steps=1000)

    cls_pred_branch = nn.Sequential(
        nn.Linear(1024, 1)
    )
    cls_pred_branch.to(accelerator.device, dtype=weight_dtype)
    num_denoising_step = 4
    denoising_timestep = diffusion.noise_steps
    denoising_step_list = torch.tensor(
        list(range(denoising_timestep - 1, 0, -(denoising_timestep // num_denoising_step))),
        dtype=torch.long,
        device=accelerator.device
    )
    timestep_interval = denoising_timestep // num_denoising_step

    min_step_percent = 0.02
    max_step_percent = 0.98
    min_step = int(min_step_percent * denoising_timestep)
    max_step = int(max_step_percent * denoising_timestep)

    gen_cls_loss_weight = 5e-3
    guidance_cls_loss_weight = 1e-2
    """build trainer"""
    train()
