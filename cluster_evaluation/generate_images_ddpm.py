import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL

# === Local imports ===
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.UNet2DWithCBM_new import UNet2DWithCBM
from classifier.celebaClassifer import DiffusersLatentClassifier
from models.CBM import CBM_new
from configs.diffusion_config import celebA
from data.DataCelebA import get_dataloader, CelebADataset
from models.diffusion_model import create_diffusion_model
import torch.nn.functional as F

# === GLOBALS & CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDPM_CKPT = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_noCBM_299.pt"
DDPM_CKPT_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_wandb_299.pt"

# FID Settings
OUTPUT_DIR = "./fid_generated_images_ddpm2"  # Folder to save images
NUM_IMAGES = 10000                      # Total images to generate (FID usually requires 2k-10k)
BATCH_SIZE = 128                        # Adjust based on your VRAM
IMAGE_SIZE = 16                       # CelebA image size

# ==============================================================
# ------------------------ FUNCTIONS ----------------------------
# ==============================================================

def load_vae():
    """Load pretrained Stable Diffusion VAE."""
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="vae"
    )
    vae = vae.to(DEVICE).eval()
    print("✅ VAE loaded.\n")
    return vae


def load_ddpm_model(config):
    """Load trained DDPM + CBM model from checkpoint."""
    print("Loading DDPM + CBM model...")
    model = UNet2DWithCBM(config, CBM_new).to(DEVICE)
    state = torch.load(DDPM_CKPT_CBM, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print("✅ DDPM model loaded successfully.\n")
    return model


def load_classic_model(config):
    """Load the standard (No-CBM) DDPM model."""
    print("Loading Classic DDPM Model...")
    model = create_diffusion_model(config).to(DEVICE)
    
    # Load checkpoint
    state = torch.load(DDPM_CKPT, map_location=DEVICE)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
        
    model.eval()
    print("✅ Classic DDPM model loaded.\n")
    return model

def save_images(images_tensor, output_dir, start_idx):
    """Convert tensor to PIL and save to disk."""
    # Convert from [-1, 1] to [0, 1]
    images_tensor = (images_tensor * 0.5 + 0.5).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    
    for i in range(images_tensor.shape[0]):
        img = to_pil(images_tensor[i].cpu())
        # Save as 00001.png, 00002.png, etc.
        save_path = os.path.join(output_dir, f"{start_idx + i:05d}.png")
        img.save(save_path)

def generate_batch(model, vae, scheduler, batch_size):
    """
    Generates a batch of images from pure Gaussian noise.
    """
    # 1. Calculate latent shape based on VAE compression (factor of 8)
    # For 128x128 image -> 16x16 latents
    latent_dim = IMAGE_SIZE  
    latent_shape = (batch_size, IMAGE_SIZE , latent_dim, latent_dim)

    # 2. Initialize random noise (x_T)
    x_t = torch.randn(latent_shape, device=DEVICE)
    
    # 3. Denoising Loop (T=1000 -> T=0)
    scheduler.set_timesteps(1000)
    
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
            # Pass return_dict=False to match your original script's behavior
            residual = model(x_t, t, return_dict=False)
            
            # If your model returns a tuple (sample, ...), take the first element
            if isinstance(residual, tuple):
                residual = residual[0]
                
            x_t = scheduler.step(residual, t, x_t).prev_sample

    # 4. Decode Latents to Pixels
    # Unscale latents before decoding (inverse of encode scaling)
    x_t = x_t / vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(x_t).sample
        
    return images

# ==============================================================
# -------------------------- MAIN -------------------------------
# ==============================================================

def main():
    # 1. Setup folders
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {OUTPUT_DIR}")
    
    # 2. Load Config & Models
    config = celebA()
    vae = load_vae()
    model = load_classic_model(config)
    
    # 3. Setup Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.config.clip_sample = False

    # 4. Generation Loop
    print(f"🚀 Starting generation of {NUM_IMAGES} images...")
    total_generated = 0
    
    # Progress bar for the total number of images
    pbar = tqdm(total=NUM_IMAGES, desc="Total Progress")
    
    while total_generated < NUM_IMAGES:
        # Determine batch size (don't over-generate on the last batch)
        current_batch_size = min(BATCH_SIZE, NUM_IMAGES - total_generated)
        
        # Generate
        images = generate_batch(model, vae, scheduler, current_batch_size)
        
        # Save
        save_images(images, OUTPUT_DIR, total_generated)
        
        # Update counters
        total_generated += current_batch_size
        pbar.update(current_batch_size)

    pbar.close()
    print(f"\n✅ Done! {total_generated} images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()