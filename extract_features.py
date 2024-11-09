import os
import torch
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as T
from diffusers.models import AutoencoderKL
from tqdm import tqdm

if __name__ == "__main__":
    image_path = "" ### Input images path
    vae_feature_path = "" ### Output features path
    vae_pretrained = "runwayml/stable-diffusion-v1-5" ### VAE model path
    scale_factor = 0.18215
    os.makedirs(vae_feature_path, exist_ok=True)

    list_image = glob(os.path.join(image_path, "*/*/*.png"))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])
    vae = AutoencoderKL.from_pretrained(vae_pretrained, subfolder="vae").to("cuda")
    vae.requires_grad_(False)

    for image_path in tqdm(list_image):
        image_name = image_path.split("/")[-1].replace(".png", "")
        folder_num = image_path.split("/")[-2]
        data_type = image_path.split("/")[-3]
        os.makedirs(os.path.join(vae_feature_path, data_type, folder_num), exist_ok=True)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.to("cuda")
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            x = vae.encode(image).latent_dist.sample() * scale_factor
        x = torch.squeeze(x, 0)
        x = x.detach().cpu().numpy()
        np.save(f'{vae_feature_path}/{data_type}/{folder_num}/{image_name}.npy', x)