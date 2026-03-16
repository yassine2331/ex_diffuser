import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL
from huggingface_hub import InferenceClient

# === Local imports ===
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.UNet2DWithCBM_new import UNet2DWithCBM
from models.CBM import CBM_new
from configs.diffusion_config import celebA
from data.DataCelebA import get_dataloader, CelebADataset
from models.diffusion_model import create_diffusion_model
import torch.nn.functional as F

# === GLOBALS ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDPM_CKPT = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_noCBM_299.pt"
IMAGE_DIR = "/home/oueslatiy/data/celeba/images"
ATTR_PATH = "/home/oueslatiy/data/celeba/list_attr_celeba.txt"
DDPM_CKPT_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_wandb_299.pt"

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
    print("✅ VAE loaded successfully.\n")
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


def load_ddpm_model_noCBM(config):
    """Load trained DDPM model from checkpoint."""
    print("Loading DDPM ...")
    model = create_diffusion_model(config).to(DEVICE)
    state = torch.load(DDPM_CKPT, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print("✅ DDPM model loaded successfully.\n")
    return model


def load_sample(config):
    """Load one CelebA sample and its attributes."""
    dataloader = get_dataloader(
        IMAGE_DIR,
        ATTR_PATH,
        selected_attrs=config.selected_attrs,
        image_size=128,
        batch_size=1,
        mode='test',
        num_workers=1
    )
    batch = next(iter(dataloader))
    real_img = batch["images"].to(DEVICE)
    attributes = batch["attributes"].float().to(DEVICE)
    print("Loaded CelebA sample with attributes:\n")
    attr_flags = [1 if i > 0 else 0 for i in attributes[0]]
    active_attrs = [config.selected_attrs[i] if attr_flags[i] == 1 else None
                    for i in range(len(attr_flags))]
    print([a for a in active_attrs if a])
    return real_img, attributes


def encode_image(vae, real_img):
    """Encode image into latent space using VAE."""
    with torch.no_grad():
        latent_dist = vae.encode(real_img).latent_dist
        z = latent_dist.sample()
        recon = vae.decode(z).sample
    print("✅ Encoding done.\n")
    return z * vae.config.scaling_factor, recon


def add_noise(scheduler, z, t_low=50, t_high=100):
    """Add noise to the latent representation."""
    t = torch.randint(t_low, t_high, (1,), device=DEVICE).long()
    noise = torch.randn_like(z)
    noisy_z = scheduler.add_noise(z, noise, t)
    return noisy_z, t


def denoise_with_cbm(model, scheduler, x_t, attributes, start_t):
    """Perform DDPM denoising with CBM interventions."""
    print("Starting DDPM denoising...")
    timesteps = scheduler.timesteps[1000 - start_t:]
    with torch.no_grad():
        for step_t in tqdm(timesteps):
            residual, _ = model(x_t, step_t, interventions=attributes, return_dict=False)
            x_t = scheduler.step(residual, step_t, x_t).prev_sample
    print("✅ Denoising done.\n")
    return x_t


def denoise(model, scheduler, x_t, start_t):
    """Perform DDPM denoising with CBM interventions."""
    print("Starting DDPM denoising...")
    timesteps = scheduler.timesteps[1000 - start_t:]
    with torch.no_grad():
        for step_t in tqdm(timesteps):
            residual = model(x_t, step_t,  return_dict=False)
            x_t = scheduler.step(residual[0], step_t, x_t).prev_sample
    print("✅ Denoising done.\n")
    return x_t


def to_pil(tensor_img):
    """Convert a tensor image ([-1,1]) to a PIL image."""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.detach().cpu()
    if tensor_img.shape[0] != 3:
        tensor_img = tensor_img.permute(2, 0, 1)
    tensor_img = (tensor_img * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)


def visualize_results(real_img, recon_vae, generated,text="generation"):
    """Display and save comparison images."""
    images = [to_pil(real_img), to_pil(recon_vae), to_pil(generated)]
    titles = ["Original", "DDPM", "DDPM + CBM"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"generation_{text}.png")
    plt.show()
    print(f"✅ Results saved as generation_{text}.png.\n")


# ==============================================================
# -------------------------- MAIN -------------------------------
# ==============================================================

def get_sample_with_attributes(image_dir, attr_path, config, target_attrs):
    """
    Iterate through the dataset until a sample matches the desired attributes.
    Keeps looping through the dataset (reshuffling at each pass).

    Args:
        image_dir (str): Path to CelebA images.
        attr_path (str): Path to CelebA attribute file.
        config: CelebA config (with selected_attrs list).
        target_attrs (dict): e.g. {"Smiling": 1, "Male": 0, "Blond_Hair": 1}

    Returns:
        (torch.Tensor, torch.Tensor): (real_img, attributes)
    """
    print(f"🔍 Searching for attributes: {target_attrs}")
    #making pulling from dataset deterministic

       
    # --- Load dataset once ---
    dataloader = get_dataloader(
        image_dir,
        attr_path,
        selected_attrs=config.selected_attrs,
        image_size=128,
        batch_size=1,
        mode='test',
        num_workers=1
    )

    # --- Precompute indices of interest ---
    target_indices = {}
    for attr, val in target_attrs.items():
        if attr not in config.selected_attrs:
            raise ValueError(f"'{attr}' not found in selected_attrs.")
        idx = config.selected_attrs.index(attr)
        target_indices[idx] = val  # index → target value

    print(f"🎯 Checking indices: {list(target_indices.keys())}")

    data_iter = iter(dataloader)
    attempts = 0

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize iterator when dataset exhausted
            print("↻ End of dataset reached — reshuffling and continuing...")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        attempts += 1
        real_img = batch["images"].to(DEVICE)
        attributes = batch["attributes"].float().to(DEVICE)

        attributes_bin = (attributes > 0).float()[0]  # convert to 0/1

        # Check only the target indices
        match = all(attributes_bin[i].item() == v for i, v in target_indices.items())

        if match:
            print(f"✅ Found matching sample after {attempts} attempts.")
            print("→ Active attributes:", 
                  [config.selected_attrs[i] for i, val in enumerate(attributes_bin) if val == 1])
            return real_img, attributes

        if attempts % 500 == 0:
            print(f"⏳ Still searching... checked {attempts} samples so far.")
###########################################################################
#################################################################################


def denoise_with_interventions(model, vae, scheduler, z, attributes, 
                               noise_start=600, noise_end=601,
                               intervention_strength=1.0,
                               interventions=None,
                               w=0.0
                               ):
    """
    Perform DDPM denoising with concept interventions and dynamic scaling.

    Args:
        model: UNet2DWithCBM
        vae: AutoencoderKL (pretrained VAE)
        scheduler: DDPMScheduler
        z (torch.Tensor): latent code from VAE encoder
        attributes (torch.Tensor): attributes tensor for conditioning
        noise_start (int): timestep for noise addition (controls noise amount)
        noise_end (int): end timestep if range desired
        intervention_strength (float): multiplier for intervention magnitude
        interventions (dict): optional concept edits {attr_index: new_value}

    Returns:
        generated (torch.Tensor): final generated image tensor
    """
    DEVICE = z.device
    t = torch.randint(noise_start, noise_end, (1,), device=DEVICE).long()
    noise = torch.randn_like(z)
    noisy_z = scheduler.add_noise(z, noise, t)

    x_t = noisy_z.clone()
    timesteps = scheduler.timesteps
    tt = 999 - t

    similarity_weights, ts, c_values = [], [], []

    print(f"\n⚙️ Starting denoising with interventions at strength {intervention_strength}...")
    print(f"Noise added at timestep {t.item()}.\n")

    with torch.no_grad():
        for step_t in tqdm(timesteps[tt:]):
            residual_un, c_un = model(x_t, step_t, return_dict=False)

            # --- Apply user-defined concept interventions ---
            if interventions is not None:
                for idx, val in interventions.items():
                    c_un[0][idx] = val 

            residual, c = model(x_t, step_t, interventions=c_un, return_dict=False)

            # --- Compute cosine similarity for adaptive scaling ---
            alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1, 1, 1, 1)
            alpha_bar_t_prev = scheduler.alphas_cumprod[step_t - 1].to(DEVICE).view(-1, 1, 1, 1) if t > 0 else torch.ones_like(alpha_bar_t)
            beta_t = scheduler.betas[step_t].to(DEVICE).view(-1, 1, 1, 1)
            residual_0 = (x_t - torch.sqrt(alpha_bar_t) * z)

            similarity = F.cosine_similarity(residual_0.flatten(1), residual.flatten(1), dim=1)
            anchor_scale = (1 + similarity[0]) / 2

            similarity_weights.append(anchor_scale.item())
            ts.append(step_t.item())
            

            # --- Dynamic blending of residuals (experimental) ---
            if 0 < step_t < 1000:
                #residual = (w + 1) * residual +  0.8*residual_0 - w * residual_un
                residual = (w + 1) * residual - w * residual_un +  (torch.norm( (w + 1) * residual - w * residual_un )/(torch.norm(residual_0) )) * residual_0 
                
            x_t = scheduler.step(residual, step_t, x_t).prev_sample

            if step_t % 100 == 0:
                print(f"Step {step_t.item()} | sim={anchor_scale.item():.3f} | c[0][2]={c[0][2].item():.3f}")
                #printing the norms of each 
                print(f"Norms | residual: {torch.norm(residual).item():.3f}, residual_0: {torch.norm(residual_0).item():.3f}, residual_un: {torch.norm(residual_un).item():.3f}")
                print(" -- -- ")
                print("norm hmm | ",1/(torch.norm(residual_0) / torch.norm(residual)))


    # --- Decode latent to image ---
    x_t = x_t / vae.config.scaling_factor
    generated = vae.decode(x_t).sample


    return generated





########################
#####################
###################

def main():
    config = celebA()
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.config.clip_sample = False

    vae = load_vae()
    model = load_ddpm_model(config)
    model_noCBM = load_ddpm_model_noCBM(config)
    
    target_attrs = {"Male": 0,"Bald":0}
    real_img, attributes = get_sample_with_attributes(IMAGE_DIR, ATTR_PATH, config, target_attrs)

    if real_img is None:
        print("No matching image found. Exiting.")
        return
    
    #real_img, attributes = load_sample(config)

    z, recon_vae = encode_image(vae, real_img)
    noisy_z, t = add_noise(scheduler, z,t_low=400, t_high=500)
    
    x_t = denoise(model_noCBM, scheduler, noisy_z.clone(), t)
    x_t = x_t / vae.config.scaling_factor
    generated_noCBM = vae.decode(x_t).sample



    x_t = denoise_with_cbm(model, scheduler, noisy_z.clone(), attributes, t)
    x_t = x_t / vae.config.scaling_factor
    generated = vae.decode(x_t).sample
    

    visualize_results(real_img, generated_noCBM, generated,text="no_intervention")

    #intervention  

    interventions = {
        10: 10.0
    }
    generated = denoise_with_interventions(
        model=model,
        vae=vae,
        scheduler=scheduler,
        z=z,
        attributes=attributes,
        noise_start=580,         # ⬅️ amount of noise added
        noise_end=601,
        intervention_strength=1.5,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        w=10.0
    )
    
    visualize_results(real_img, recon_vae, generated,"intervention_1")

    ############
    interventions = {
        10: 1.0
    }
    generated = denoise_with_interventions(
        model=model,
        vae=vae,
        scheduler=scheduler,
        z=z,
        attributes=attributes,
        noise_start=580,         # ⬅️ amount of noise added
        noise_end=601,
        intervention_strength=1.5,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        w=10.0
    )
    visualize_results(real_img, recon_vae, generated,"intervention_2")
    
    ############

    interventions = {
        
        10: 5.0
    }
    generated = denoise_with_interventions(
        model=model,
        vae=vae,
        scheduler=scheduler,
        z=z,
        attributes=attributes,
        noise_start=580,         # ⬅️ amount of noise added
        noise_end=601,
        intervention_strength=1.5,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        w=5.0
    )
    visualize_results(real_img, recon_vae, generated,"intervention_3")



if __name__ == "__main__":
    main()
