import os
import cv2
import torch
import torchvision
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from models.diffusion import Diffusion
from data_loader.HWT_dataset import ContentData
from utils.checkpoint import load_checkpoint
from models.builder import build_model
from utils.misc import read_config

import warnings

warnings.filterwarnings("ignore")  # ignore warning


def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    im = torchvision.transforms.ToPILImage()(grid)
    im.save(path)
    return im


def get_style_ref_random_style(style_path, laplace_path):
    batch = []
    style_len = 352
    style_image = cv2.imread(style_path, flags=0)
    style_image_copy = style_image.copy()
    laplace_image = cv2.imread(laplace_path, flags=0)
    style_image = style_image / 255.0
    laplace_image = laplace_image / 255.0
    style_ref = torch.from_numpy(style_image).unsqueeze(0)
    style_ref = style_ref.to(torch.float32)
    laplace_ref = torch.from_numpy(laplace_image).unsqueeze(0)
    laplace_ref = laplace_ref.to(torch.float32)
    batch.append({'style': style_ref, 'laplace': laplace_ref})
    s_width = [item['style'].shape[2] for item in batch]
    if max(s_width) < style_len:
        max_s_width = max(s_width)
    else:
        max_s_width = style_len
    style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width],
                           dtype=torch.float32)
    laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width],
                              dtype=torch.float32)
    for idx, item in enumerate(batch):
        try:
            if max_s_width < style_len:
                style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
            else:
                style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :style_len]
                laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :style_len]

        except:
            print('style', item['style'].shape)
    return style_ref, laplace_ref, style_image_copy


def get_image(image_path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transforms(image)
    image = image.unsqueeze(0)
    return image


def infer_text(texts, image_path, style_image_path, laplace_image_path, config, work_dir=''):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(texts, str):
        texts = [texts]

    vae = None

    """ MODEL """
    #####
    model_kwargs = {"qk_norm": config.qk_norm, "differential": config.differential}
    model = build_model(config.model,
                        config.grad_checkpointing,
                        **model_kwargs).train()

    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), resume_optimizer=False,
            resume_lr_scheduler=False)
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')

    diffusion = Diffusion(device=device, noise_offset=config.noise_offset)
    model = model.to(device).eval()

    '''Dataset'''
    load_content = ContentData()
    print("Running infer... ")
    # prepare input
    if vae is None:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, subfolder="vae", torch_dtype=torch.float16).to(
            device)
        vae.requires_grad_(False)
    torch.cuda.empty_cache()
    style_ref, laplace_ref, _ = get_style_ref_random_style(style_image_path, laplace_image_path)
    images = get_image(image_path)
    images, style_ref, laplace_ref = images.to(device), style_ref.to(device), laplace_ref.to(device)
    for text in texts:
        torch.cuda.empty_cache()
        text_ref = load_content.get_content(text)
        text_ref = text_ref.to(device).repeat(style_ref.shape[0], 1, 1, 1)
        x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(device)
        preds = diffusion.ddim_sample(model, vae, style_ref.shape[0], x, style_ref, laplace_ref, text_ref,
                                      config=config, sampling_timesteps=50, eta=0)
        out_path = os.path.join(os.path.join(work_dir, "save_sample_dir"), f"{text}.png")
        if not os.path.exists(str(Path(out_path).parent)):
            print("here")
            os.makedirs(str(Path(out_path).parent), exist_ok=True)
        save_images(preds, out_path)

    del vae
    del images, style_ref, laplace_ref, text_ref, preds


def infer_page(texts, style_image_path, laplace_image_path, config, work_dir=''):
    alphabet = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    texts = texts.replace('\n', ' ')
    texts = texts.replace('\n', ' ')
    texts = ''.join(c for c in texts if c in alphabet)  # just to avoid problems with the font dataset
    texts = [word for word in texts.split()]
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = None

    """ MODEL """
    #####
    model_kwargs = {"qk_norm": config.qk_norm, "differential": config.differential}
    model = build_model(config.model,
                        config.grad_checkpointing,
                        **model_kwargs).train()

    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), resume_optimizer=False,
            resume_lr_scheduler=False)
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')

    diffusion = Diffusion(device=device, noise_offset=config.noise_offset)
    model = model.to(device).eval()

    '''Dataset'''
    load_content = ContentData()
    # prepare input
    if vae is None:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, subfolder="vae", torch_dtype=torch.float16).to(
            device)
        vae.requires_grad_(False)
    torch.cuda.empty_cache()
    all_page1 = []
    all_page2 = []
    for i in range(len(style_image_path)):
        print(f"style image {i + 1}/{len(style_image_path)}")
        style_ref, laplace_ref, style_image_copy = get_style_ref_random_style(style_image_path[i],
                                                                              laplace_image_path[i])
        style_ref, laplace_ref = style_ref.to(device), laplace_ref.to(device)
        fake_pred = []
        for text in tqdm(texts, desc='gen text'):
            torch.cuda.empty_cache()
            text_ref = load_content.get_content(text)
            text_ref = text_ref.to(device).repeat(style_ref.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(device)
            preds = diffusion.ddim_sample(model, vae, style_ref.shape[0], x, style_ref, laplace_ref, text_ref,
                                          config=config, sampling_timesteps=50, eta=0)
            grid = torchvision.utils.make_grid(preds)
            im = torchvision.transforms.ToPILImage()(grid)
            im_np = np.array(im)
            im_cv = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
            fake_pred.append(im_cv)

        word_t = []
        word_l = []
        gap = np.ones([64, 16, 3]) * 255
        line_wids = []
        rwidth = 980
        for idx, fake_ in enumerate(fake_pred):
            word_t.append(fake_)
            word_t.append(gap)

            if sum(t.shape[-2] for t in word_t) >= rwidth or idx == len(fake_pred) - 1:
                line_ = np.concatenate(word_t, axis=1)
                word_l.append(line_)
                line_wids.append(line_.shape[1])
                word_t = []

        gap_h = np.ones([16, max(line_wids), 3]) * 255
        page_ = []
        for l in word_l:
            pad_ = np.ones([64, max(line_wids) - l.shape[1], 3]) * 255
            page_.append(np.concatenate([l, pad_], 1))
            page_.append(gap_h)

        page1 = np.concatenate(page_, 0)
        word_t = []
        word_l = []
        line_wids = []

        gap = np.ones([style_image_copy.shape[0], 16]) * 255
        word_t.append(style_image_copy)
        word_t.append(gap)
        line_ = np.concatenate(word_t, -1)
        word_l.append(line_)
        line_wids.append(line_.shape[1])
        word_t = []
        gap_h = np.ones([16, max(line_wids)]) * 255
        page_ = []

        for l in word_l:
            page_.append(l)
            page_.append(gap_h)
        page2 = np.concatenate(page_, 0)
        merge_w_size = max(page1.shape[0], page2.shape[0])
        if page1.shape[0] != merge_w_size:
            page1 = np.concatenate([page1, np.ones([merge_w_size - page1.shape[0], page1.shape[1]]) * 255], 0)

        if page2.shape[0] != merge_w_size:
            page2 = np.concatenate([page2, np.ones([merge_w_size - page2.shape[0], page2.shape[1]]) * 255], 0)

        page2 = np.repeat(page2[:, :, np.newaxis], 3, axis=2)
        all_page1.append(page1)
        all_page2.append(page2)

    if len(all_page1) == 1:
        cv2.imwrite("2page_check.jpg", np.concatenate([all_page2[0], all_page1[0]], 1))
    else:
        page1s_ = np.concatenate(all_page1, 0)
        max_wid = max([i.shape[1] for i in all_page2])
        padded_page2s = []

        for para in all_page2:
            padded_page2s.append(np.concatenate([para, np.ones([para.shape[0], max_wid - para.shape[1], 3]) * 255], 1))

        padded_page2s_ = np.concatenate(padded_page2s, 0)
        cv2.imwrite("2page_check_again.jpg", np.concatenate([padded_page2s_, page1s_], 1))


if __name__ == "__main__":
    ### infer text
    config_path = ''
    configs = read_config(config_path)
    work_dir = ''

    text = 'Bella'
    image_path = ''
    style_im_path = ''
    laplace_im_path = ''
    infer_text(text, image_path, style_im_path, laplace_im_path, configs, work_dir)

    ### infer page
    # config_path = ''
    # configs = read_config(config_path)
    # work_dir = ''

    # text = "The well-known story I told at the about in L.A./ California, New York,...and Richmond went as follows: It amused people who knew Tommy to hear this; however, it distressed Suzi when Tommy (1982--2019) asked, 'How can I find out who yelled*, 'FIRE!' in the theater? ZOE DESCANEL.' #PANIC. is 374"
    # image_path = ''
    # style_im_path = ['',
    #                  '']
    # laplace_im_path = ['',
    #                    '']

    # infer_page(text, style_im_path, laplace_im_path, configs, work_dir)
