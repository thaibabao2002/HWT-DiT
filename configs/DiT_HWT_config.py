# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'bf16'
multi_scale = True

load_from = None
resume_from = None

### README: load_from = model_checkpoint_path; resume_from = dict(checkpoint="model_checkpoint_path", ...)
# load_from = ""
# resume_from = dict(
#         checkpoint="",
#         load_ema=False,
#         resume_optimizer=True,
#         resume_lr_scheduler=True
#     )
qk_norm = True # Use qk normalize
differential = True # Use differential transformer
vae_pretrained = "runwayml/stable-diffusion-v1-5"  # sd1.5 vae
feat_model = "model_zoo/RN18_class_10400.pth"
scale_factor = 0.18215  # ldm vae: 0.18215; sdxl vae: 0.13025
noise_offset = 0
load_ocr = False
ocr_model = "model_zoo/vae_HTR138.pth"

# training setting
seed = 43
skip_step = 0
num_workers = 8
train_batch_size = 64
test_batch_size = 64
num_epochs = 1000
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
gc_step = 1
optimizer = dict(type='CAMEWrapper', lr=5e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
## optimizer = dict(type='AdamW', lr=1e-4, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)
lr_schedule = 'constant'
visualize = True
log_interval = 100
start_save_epochs = 10
save_model_epochs = 20
save_model_steps = 10000
eval_sampling_steps = 1000
work_dir = 'output/HWDiT'
debug = False
valid_num=0  # take as valid aspect-ratio when sample number >= valid_num

# data
load_vae_feat = False
vae_path = "/home/baotb/Desktop/D/BaoTB/code/One-DM/data/IAM64_feature_64_32x_3_channel"
IMAGE_PATH = "/home/baotb/Desktop/D/BaoTB/code/One-DM/data/IAM_train_test_64_32x_3_channel"
STYLE_PATH = "/home/baotb/Desktop/D/BaoTB/code/One-DM/data/IAM_train_test_64_32x_3_channel"
LAPLACE_PATH = "/home/baotb/Desktop/D/BaoTB/code/One-DM/data/IAM_train_test_64_32x_laplace_3_channel"