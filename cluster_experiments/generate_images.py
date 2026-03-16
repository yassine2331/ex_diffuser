import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from diffusers import DDPMScheduler
from torchvision.utils import make_grid
 
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
 
from models.UNet2DWithCBM_new import UNet2DWithCBM
from models.CBM import CBM_new
from configs.diffusion_config import celebA
from data.DataCelebA import get_dataloader,CelebADataset
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../VAE')))
 

from pathlib import Path
import yaml


import torchvision.transforms.functional as TF
import torchvision
from diffusers.models import AutoencoderKL
from PIL import Image


# hugging face interface client
from huggingface_hub import InferenceClient

# --- Add local project root to sys.path (for configs, models, etc.) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

# --- Now import safely ---
                    # From VAE
from configs.diffusion_config import celebA         # From ex_diffuser/configs
from models.UNet2DWithCBM_new import UNet2DWithCBM  # From ex_diffuser/models
from models.CBM import CBM_new                      # From ex_diffuser/models
from data.DataCelebA import get_dataloader
from train.train_celeba import train_loop
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup



# Paths
ddpm_ckpt = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_wandb_299.pt"

image_dir = "/home/oueslatiy/data/celeba/images"
attr_path = "/home/oueslatiy/data/celeba/list_attr_celeba.txt"
    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = celebA()
 
# --- Load VAE ---
client = InferenceClient(
        provider="fal-ai",
        api_key="###",
    )


#---------vae ----------   
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    subfolder="vae"
)
#-----------------------
# tryin a diffrent vae 


"""vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae"
)"""


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
vae.eval()

print("Loading the vae is done... \n")

#modelll 
 
# --- Load DDPM + CBM ---
model = UNet2DWithCBM(config, CBM_new).to(DEVICE)
model.load_state_dict(torch.load(ddpm_ckpt, map_location=DEVICE)['model_state_dict'])
model.eval()
 
# --- Load CelebA sample ---
dataloader = get_dataloader(image_dir, attr_path, selected_attrs=config.selected_attrs, image_size=128, batch_size=1, mode='test', num_workers=1)
batch = next(iter(dataloader))
real_img = batch["images"].to(DEVICE)
attributes = batch["attributes"].float().to(DEVICE)  # for intervention
data = CelebADataset(image_dir,attr_path,config.selected_attrs)

attributes2 = [ 1 if i > 0 else 0 for i in attributes[0] ]
#print(config.selected_attrs[attributes])

print([config.selected_attrs[i] if attributes2[i] == 1 else 0 for i in range(len(attributes2))])

# --- Encode image ---
with torch.no_grad():
    latent_dist= vae.encode(real_img).latent_dist
    z = latent_dist.sample() 
    recon_vae = vae.decode(z).sample

print("BEFORE SCALING ")
print("Encoded latent shape: ", z.shape)
print("std ", torch.std(z))
print("mean ", torch.mean(z))
print("max ", torch.max(z))
print("min ", torch.min(z))
print(" -- -- ")
# --- Add noise ---

z = z * vae.config.scaling_factor
# --- Evaluate function (from evaluate_celeba.py) ---
print("Encoded latent shape: ", z.shape)
print("std ", torch.std(z))
print("mean ", torch.mean(z))
print("max ", torch.max(z))
print("min ", torch.min(z))
print(" -- -- ")
# --- Add noise ---
scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.config.clip_sample = False  # disable clipping for generation
t = torch.randint(50, 100, (1,), device=DEVICE).long()  # mild noise
#t = torch.tensor(999)
noise = torch.randn_like(z)


#z = z * 0.18215  # scaling factor for stable diffusion v1.5


noisy_z = scheduler.add_noise(z, noise, t)
 
# --- DDPM Denoising with Interventions ---
x_t = noisy_z.clone()
timesteps = scheduler.timesteps[1000 - t :]
#x_t = x_t * vae.config.scaling_factor

with torch.no_grad():
    for step_t in tqdm(timesteps):
        residual, _ = model(x_t, step_t, interventions=attributes, return_dict=False)
        x_t = scheduler.step(residual, step_t, x_t).prev_sample
        if step_t % 100 == 0:
            print("thresholing ", scheduler.config.clip_sample)
            print("Step ############## ", step_t)
            #images = x_t / 0.18215
            #images = x_t / vae.config.scaling_factor 
            images = x_t #* 1.25
            print("x_t ", x_t.shape)
            print("std ", torch.std(x_t))
            print("mean ", torch.mean(x_t))
            print("max ", torch.max(x_t))
            print("min ", torch.min(x_t))
            print(vae.config.scaling_factor )
            #mu, logvar = vae.encode(clean_images)
            #z = vae.reparameterize(mu, logvar)
            print("EVAL ", images.shape)

        
        

print("Step ############## ", step_t)
#images = x_t / 0.18215
#images = x_t / vae.config.scaling_factor 
images = x_t #* 1.25
print("x_t ", x_t.shape)
print("std ", torch.std(x_t))
print("mean ", torch.mean(x_t))
print("max ", torch.max(x_t))
print("min ", torch.min(x_t))
print(vae.config.scaling_factor )
#mu, logvar = vae.encode(clean_images)
#z = vae.reparameterize(mu, logvar)
print("EVAL ", images.shape)

images = images / vae.config.scaling_factor

generated = vae.decode(images).sample


    
# --- Post-process and Show ---
def to_pil(tensor_img):
    # remove batch dimension
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)
    
    # move to cpu
    tensor_img = tensor_img.detach().cpu()

    # ensure channel-first shape
    if tensor_img.shape[0] != 3:
        tensor_img = tensor_img.permute(2, 0, 1)

    # normalize from [-1,1] to [0,1]
    tensor_img = (tensor_img * 0.5 + 0.5).clamp(0, 1) 

    return transforms.ToPILImage()(tensor_img)

images = [to_pil(real_img), to_pil(recon_vae), to_pil(generated)]
titles = ["Original", "VAE Reconstruction", "DDPM + CBM"]
 
# --- Display ---
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(axs, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig("celeba_generation.png")
