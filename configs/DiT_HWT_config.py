_base_ = ['./PixArt_xl2_internal.py']
# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'fp16'
fp32_attention = True
load_from = "/home/baotb/Desktop/D/BaoTB/code/One-DM/output/IAM-spatial_transformer_continue2/checkpoints/epoch_60_step_29477.pth"
# load_from = None
resume_from = None
# vae_pretrained = "madebyollin/sdxl-vae-fp16-fix"  # sdxl vae
vae_pretrained = "runwayml/stable-diffusion-v1-5"  # sdxl vae
feat_model = "model_zoo/RN18_class_10400.pth"
multi_scale = True

# training setting
num_workers = 8
train_batch_size = 128
test_batch_size = 64
num_epochs = 500
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=5e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=20000)
load_vae_feat = True
vae_path = "./data/IAM64_feature"

visualize = True
log_interval = 100
start_save_epochs = 10
save_model_epochs = 20
save_model_steps = 10000
eval_sampling_steps = 1000
work_dir = 'output/IAM-spatial_transformer'
debug = False

load_ocr = False
ocr_model = "model_zoo/vae_HTR138.pth"

# data
IMAGE_PATH = "./data/IAM64-new"
STYLE_PATH = "./data/IAM64-new"
LAPLACE_PATH = "./data/IAM64_laplace"

# pixart-sigma
# scale_factor = 0.13025
scale_factor = 0.18215
class_dropout_prob = 0.1
kv_compress = False
kv_compress_config = {
    'sampling': 'conv',
    'scale_factor': 2,
    'kv_compress_layer': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
}
qk_norm = False
noise_offset = 0