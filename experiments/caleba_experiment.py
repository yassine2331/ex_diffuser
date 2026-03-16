import sys
import os
from pathlib import Path
import yaml
import torch

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

def run():

    client = InferenceClient(
        provider="fal-ai",
        api_key="#####",
    )


    #---------vae ----------   
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="vae"
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    vae.eval()

    print("Loading the vae is done... \n")

    #modelll 

    model_name = sys.argv[1]

    image_dir = "/home/oueslatiy/data/celeba/images"
    attr_path = "/home/oueslatiy/data/celeba/list_attr_celeba.txt"
    
    config = celebA()
    selected_attrs =config.selected_attrs
    batch_size = config.train_batch_size
    dataloader = get_dataloader(image_dir, attr_path, selected_attrs=selected_attrs, image_size=128, batch_size=batch_size, mode='train', num_workers=4)
    model = UNet2DWithCBM(config,CBM_new)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler,model_name=model_name,vae=vae)

if __name__ == "__main__":
    run()