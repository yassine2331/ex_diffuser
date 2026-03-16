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
from classifier.celebaClassifer import DiffusersLatentClassifier
from models.CBM import CBM_new
from configs.diffusion_config import celebA
from data.DataCelebA import get_dataloader, CelebADataset
from models.diffusion_model import create_diffusion_model
import torch.nn.functional as F
import imageio

# === GLOBALS ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDPM_CKPT = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_noCBM_299.pt"
IMAGE_DIR = "/home/oueslatiy/data/celeba/images"
ATTR_PATH = "/home/oueslatiy/data/celeba/list_attr_celeba.txt"
DDPM_CKPT_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_wandb_299.pt"
CLASSIFIER = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/classifier_celeba_final_best.pt"

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

def load_classifier(config):
    """Load trained CelebA classifier from checkpoint."""
    print("Loading CelebA classifier...")
    classifier = DiffusersLatentClassifier(input_channels=16, selected_attributes=config.selected_attrs).to(DEVICE)
    state = torch.load(CLASSIFIER, map_location=DEVICE)
    classifier.load_state_dict(state)
    classifier.eval()
    print("✅ Classifier loaded successfully.\n")
    return classifier

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
    
    masks = []
    mask_o = torch.zeros_like(noisy_z)  # initialize the mask outside the loop
    n = 0 
    with torch.no_grad():
        for step_t in tqdm(timesteps[tt:]):
            residual_un, c_un = model(x_t, step_t, return_dict=False)

            # --- Apply user-defined concept interventions ---
            if interventions is not None:
                for idx, val in interventions.items():
                    c_un[0][idx] = val*intervention_strength
                    attributes[0][idx] = val*intervention_strength
            residual, c = model(x_t, step_t, interventions=attributes, return_dict=False)

            # --- Compute cosine similarity for adaptive scaling ---
            alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1, 1, 1, 1)
            alpha_bar_t_prev = scheduler.alphas_cumprod[step_t - 1].to(DEVICE).view(-1, 1, 1, 1) if t > 0 else torch.ones_like(alpha_bar_t)
            beta_t = scheduler.betas[step_t].to(DEVICE).view(-1, 1, 1, 1)
            residual_0 = (x_t - torch.sqrt(alpha_bar_t) * z)/torch.sqrt(1 - alpha_bar_t)

            similarity = F.cosine_similarity(residual_0.flatten(1), residual.flatten(1), dim=1)
            anchor_scale = (1 + similarity[0]) / 2

            similarity_weights.append(anchor_scale.item())
            ts.append(step_t.item())
            
            if 0 < step_t < 1000:
                
        
                diffmap = torch.abs(((w+1) * residual - w * residual_un ) -  residual_0)
                mask = diffmap / torch.max(diffmap)
                #bineraize the mask
                
                #apply gaussian blur to the mask to smooth it
                
                threshold = 0.3
                mask = (mask > threshold).float()
               
                mask_o = transforms.functional.gaussian_blur(mask, kernel_size=(3,3), sigma=(5.0,5.0))
                #mask_o = (mask_o * n  + mask) / (n + 1)
                #n += 1
                #mask_o = torch.clamp(mask_o, 0, 1)

                
            # --- Dynamic blending of residuals (experimental) ---
            if 60 < step_t < 1000:
                #residual = (w + 1) * residual +  0.7*residual_0 - w * residual_un
            

                residual = mask_o * ((w + 1) * residual - w * residual_un )+  (1 - mask_o)* residual_0 # normalization *(torch.norm( (w + 1) * residual - w * residual_un )/(torch.norm(residual_0) ))  
                
                mask_ing = vae.decode(mask_o / vae.config.scaling_factor).sample

                masks.append(mask_ing.detach().cpu().numpy())
            else:
             
        
                residual =  ((w + 1) * residual - w * residual_un )
                #residual  = ((w+1) * residual - w * residual_un ) 
            
            # empolyinga a maskign strategy base on the predicted noise 

          
            #residual = mask * residual +  * residual_0

            x_t = scheduler.step(residual, step_t, x_t).prev_sample

            if step_t % 100 == 0:
                print(f"Step {step_t.item()} | sim={anchor_scale.item():.3f} | c[0][2]={c[0][2].item():.3f}")

                #printing the norms of each 
                print(f"Norms | residual: {torch.norm(residual).item():.3f}, residual_0: {torch.norm(residual_0).item():.3f}, residual_un: {torch.norm(residual_un).item():.3f}")
                print(" -- -- ")
                print("norm hmm | ",1/(torch.norm(residual_0) / torch.norm(residual)))


    # --- Decode latent to image ---
    z = x_t.clone()
    x_t = x_t / vae.config.scaling_factor
    generated = vae.decode(x_t).sample
    # creating a gif animation for all the mask values 
    print("shape of masks ", len(masks), masks[0].shape)
    print("shape of generated ", generated.shape)
    # decoding the masks and the
    # first we average all chanels of the masks they have shape of (1 , 3 , 128 , 128)
    masks  = [np.mean(mask, axis=1, keepdims=True) for mask in masks]
    gif_path = "masks_animation.gif"
    imageio.mimsave(gif_path, [ (mask[0,0] * 255).astype(np.uint8) for mask in masks], duration=0.1)
    print(f"Saved mask animation to {gif_path}")

    return generated,z


def normalize_grad(g,e):
    flat = g.flatten(1)
    e_flat = e.flatten(1)
    coef= (e_flat.norm(p=2, dim=1, keepdim=True).view(-1,1,1,1) ) / (flat.norm(p=2, dim=1, keepdim=True).view(-1,1,1,1) + 1e-6)
    return g * coef


def denoise_with_interventions_baseline_(model, vae,classifier, scheduler, z, attributes, 
                               noise_start=600, noise_end=601,
                               intervention_strength=1.0,
                               interventions=None,
                               distance_strength=1.0
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
            residual_un = model(x_t, step_t, return_dict=False)[0]

            alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1,1,1,1)
            sqrt_1m_ab  = torch.sqrt(1.0 - alpha_bar_t)
            variance_t = scheduler._get_variance(step_t).to(DEVICE)
        

            x_est = (x_t - sqrt_1m_ab * residual_un) / torch.sqrt(alpha_bar_t)

            #creating to class vector 
            # classifier guidance on x_est toward to_class
            logits = classifier(x_est)
            
            total_loss = 0.0
            has_loss = False
            
            for idx, target_val in interventions.items():
                target_tensor = torch.tensor([target_val], device=DEVICE, dtype=torch.float)
                # We want to minimize BCE (make prediction match target)
                print("target_tensor ", target_tensor)
                loss_cls = F.binary_cross_entropy_with_logits(logits[:, idx], target_tensor)
                total_loss += loss_cls
                has_loss = True

            print("logits ", logits)
            print("attributes ", attributes)
            print("total loss ", total_loss)

            
            grad_cls = torch.autograd.grad(total_loss, x_est, create_graph=False)[0]

            # L1 distance guidance w.r.t. clean image
            l1_loss = F.l1_loss(x_est, z, reduction='sum')
            grad_dist = torch.autograd.grad(l1_loss, x_est, create_graph=False)[0]

            # balance grads
            grad_cls  = normalize_grad(grad_cls,  residual_un)
            grad_dist = normalize_grad(grad_dist, residual_un)
            g_total   = intervention_strength * grad_cls - distance_strength * grad_dist

            # epsilon_hat and step
            #epsilon_hat = residual_un - sqrt_1m_ab * g_total
            epsilon_hat = residual_un + variance_t * g_total
            print ("variance_t ", variance_t)
            step = scheduler.step(epsilon_hat, step_t, x_t)
            x_t = step.prev_sample

        logits = classifier(x_t)
        print("Final logits after denoising: ", logits)
            
    # --- Decode latent to image ---
    x_t = x_t / vae.config.scaling_factor
    generated = vae.decode(x_t).sample


    return generated


####



def denoise_with_interventions_baseline(
    model, 
    vae, 
    classifier, 
    scheduler, 
    z, 
    attributes, 
    noise_start=600, 
    noise_end=601,
    intervention_strength=1.0,
    interventions=None,
    distance_strength=1.0
):
    """
    Perform DDPM denoising with concept interventions and dynamic scaling.
    Fixed version: Solves 'element 0 of tensors does not require grad'.
    """
    DEVICE = z.device
    
    # --- Helper: Normalize Gradient ---
    """def normalize_grad(grad, ref):
        norm = torch.norm(grad)
        return grad / (norm + 1e-8) * torch.norm(ref)
    """
    # 1. Initialize State
    t = torch.randint(noise_start, noise_end, (1,), device=DEVICE).long()
    noise = torch.randn_like(z)
    noisy_z = scheduler.add_noise(z, noise, t)

    x_t = noisy_z.clone()
    
    # 2. Setup Timesteps
    timesteps = scheduler.timesteps
    # Find the index corresponding to our noise start 't'
    start_index = (timesteps == t.item()).nonzero(as_tuple=True)[0].item()
    inference_timesteps = timesteps[start_index:]

    print(f"\n⚙️ Starting denoising with interventions at strength {intervention_strength}...")

    # 3. Denoising Loop
    for step_t in tqdm(inference_timesteps):
        
        # --- A. UNet Prediction (No Grad Needed) ---
        with torch.no_grad():
            output = model(x_t, step_t, return_dict=False)
            # Handle tuple return (residual, concepts) vs single tensor
            residual_un = output[0] if isinstance(output, tuple) else output

        # --- B. Gradient Calculation Setup ---
        # FIX: We must enable gradients on the input to the guidance calculation
        x_in = x_t.detach().requires_grad_(True)

        # Precompute scalars for x_est formula
        alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1, 1, 1, 1)
        sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar_t)
        sqrt_ab = torch.sqrt(alpha_bar_t)
        variance_t = scheduler._get_variance(step_t).to(DEVICE)
        # Re-calculate x_est using x_in so the graph is connected
        # x_est = (x_t - sqrt(1-ab) * epsilon) / sqrt(ab)
        x_est = (x_in - sqrt_1m_ab * residual_un.detach()) / sqrt_ab

        # --- C. Classifier Logic ---
        logits = classifier(x_est)
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        if interventions is not None:
            for idx, target_val in interventions.items():
                target_tensor = torch.tensor([target_val], device=DEVICE, dtype=torch.float)
                if step_t == 5:
                    print("target_tensor ", target_tensor)
                    print("logits[:, idx] ", logits[:, idx])
                # Sum BCE losses
                loss_cls = F.binary_cross_entropy_with_logits(logits[:, idx], target_tensor)
                total_loss += loss_cls

        # --- D. Compute Gradients Separately (Your Logic) ---
        
        # 1. Gradient of Classifier Loss
        if total_loss.item() != 0:
            # FIX: We take grad w.r.t x_in (which is x_t), NOT x_est directly.
            # (Mathematically equivalent but fixes the RuntimeError)
   
            grad_cls = torch.autograd.grad(total_loss, x_est, retain_graph=True)[0]
        else:
            grad_cls = torch.zeros_like(x_est)

        # 2. Gradient of Distance Loss (L2)
        l2_loss = F.mse_loss(x_est, z, reduction='sum')
        p=1.5 # Lp norm degree
        lp_loss = torch.sum(torch.pow(torch.abs(x_est - z), p))
        # FIX: We take grad w.r.t x_in
        grad_dist = torch.autograd.grad(lp_loss, x_in)[0]

        # --- E. Balance and Combine Gradients (Your Logic) ---
        
        # Normalize individually
        grad_cls = normalize_grad(grad_cls, residual_un)
        grad_dist = normalize_grad(grad_dist, residual_un)

        # Combine using your specific formula:
        # Note: Usually minimizing loss means subtracting gradient.
        # Your formula: (Strength * Class) - (Distance_Strength * Distance)
        g_total = (intervention_strength * grad_cls) - (distance_strength * grad_dist)

        # --- F. Update Step ---
        # Modify the noise prediction
        #epsilon_hat = residual_un - sqrt_1m_ab * g_total
        epsilon_hat = residual_un + variance_t * g_total
        print("sqrt_1m_ab ", sqrt_1m_ab)
        print ("variance_t ", variance_t)
        print("diffrence ", sqrt_1m_ab - variance_t)
        # Step with Scheduler
        x_t = x_t.detach()
        step_output = scheduler.step(epsilon_hat, step_t, x_t)
        x_t = step_output.prev_sample
    
    logits = classifier(x_t)
    print("Final logits after denoising: ", logits)
    print("verification :", logits[:,0])
            
    # --- Decode ---
    x_t = x_t / vae.config.scaling_factor
    with torch.no_grad():
        generated = vae.decode(x_t).sample

    return generated


import torch.nn.functional as F
import torch
from tqdm import tqdm
















def unit_normalize(g, eps=1e-8):
    """Normalizes a tensor to have an L2 norm of 1 per image."""
    flat = g.flatten(1)
    norm = flat.norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
    return g / (norm + eps)

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import imageio
from torchvision import transforms

def unit_normalize(g, eps=1e-8):
    """Normalizes a tensor to have an L2 norm of 1 per image."""
    flat = g.flatten(1)
    norm = flat.norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
    return g / (norm + eps)

def denoise_with_interventions_new(model, vae, scheduler, z, attributes, 
                               noise_start=600, noise_end=601,
                               intervention_strength=1.0,
                               distance_strength=0.1, # Added param
                               interventions=None,
                               w=0.0,
                               classifier=None):
    """
    Perform DDPM denoising with concept interventions, dynamic scaling, 
    and dual-gradient guidance (Classifier + Distance).
    """
    DEVICE = z.device
    t = torch.randint(noise_start, noise_end, (1,), device=DEVICE).long()
    noise = torch.randn_like(z)
    noisy_z = scheduler.add_noise(z, noise, t)

    x_t = noisy_z.clone()
    timesteps = scheduler.timesteps
    tt = 999 - t

    similarity_weights, ts, c_values = [], [], []
    masks = []
    mask_o = torch.zeros_like(noisy_z)

    print(f"\n⚙️ Starting denoising at strength {intervention_strength}...")
    
    # We use a loop where some parts require grad and some don't
    for step_t in tqdm(timesteps[tt:]):
        # 1. Prediction Pass (No Grad to save memory/prevent UNet backprop)
        with torch.no_grad():
            residual_un, c_un = model(x_t, step_t, return_dict=False)

            if interventions is not None:
                for idx, val in interventions.items():
                    c_un[0][idx] = val * intervention_strength
                    attributes[0][idx] = val * intervention_strength
            
            residual, c = model(x_t, step_t, interventions=attributes, return_dict=False)

        # 2. Gradient Guidance Section (Enable Grad)
        # We wrap this in a context only for the guidance calculation
        x_in = x_t.detach().requires_grad_(True)
        
        # Re-derive x_est (fdn) inside the graph to maintain path to x_in
        alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1, 1, 1, 1)
        sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar_t)
        sqrt_ab = torch.sqrt(alpha_bar_t)
        
        # Key: x_est is now linked to x_in
        x_est = (x_in - sqrt_1m_ab * residual_un) / sqrt_ab

        # --- C. Loss Calculations ---
        # 1. Classifier Guidance
        grad_cls = torch.zeros_like(x_in)
        if classifier is not None and interventions is not None:
            logits = classifier(x_est)
            total_cls_loss = torch.tensor(0.0, device=DEVICE)
            for idx, target_val in interventions.items():
                target_tensor = torch.full((logits.size(0),), float(target_val), device=DEVICE)
                total_cls_loss += F.binary_cross_entropy_with_logits(logits[:, idx], target_tensor)
            
            grad_cls = torch.autograd.grad(total_cls_loss, x_in, retain_graph=True)[0]

        # 2. Distance Guidance
        dist_loss = F.l1_loss(x_est, z, reduction='sum')
        grad_dist = torch.autograd.grad(dist_loss, x_in)[0]

        # --- E. Construct g_update ---
        g_cls_unit = unit_normalize(grad_cls)
        g_dist_unit = unit_normalize(grad_dist)
        
        # Update direction: move away from loss (gradient descent)
        g_update = (-0.1 * g_cls_unit) - (0.15 * g_dist_unit)

        # Apply guidance to x_t (Langevin step)
        # Note: You can adjust the step size here if needed
        #with torch.no_grad():
            #x_t = x_t + g_update 

        # 3. Masking & Blending Logic (No Grad)
        with torch.no_grad():
            # Adaptive scaling calculations
            residual_0 = (x_t - torch.sqrt(alpha_bar_t) * z) / torch.sqrt(1 - alpha_bar_t)
            similarity = F.cosine_similarity(residual_0.flatten(1), residual.flatten(1), dim=1)
            anchor_scale = (1 + similarity[0]) / 2
            
            if 0 < step_t < 1000:
                alpha = 1
                # diffmap = torch.abs(((w + 1) * residual - w * residual_un) - residual_0)
                # mask = diffmap / (torch.max(diffmap) + 1e-8)
                mask = g_cls_unit.abs()
                mask = mask / (torch.max(mask) + 1e-8)
                #mask = transforms.functional.gaussian_blur(mask, kernel_size=(3, 3), sigma=(1.0, 1.0))
                threshold = 0.1
                mask = (mask > threshold).float()
                mask_o = mask_o * (1- alpha) + mask * alpha  
                #mask_o = transforms.functional.gaussian_blur(mask_o, kernel_size=(3, 3), sigma=(1.0, 1.0))

            if 0 < step_t < 1000:
                residual = mask_o * ((w + 1) * residual - w * residual_un) + (1 - mask_o) * residual_0
                # Storing gradient units for the GIF as requested
                mask_ing = vae.decode(mask_o / vae.config.scaling_factor).sample
                masks.append(mask_ing.detach().cpu().numpy())
            else:
                residual = ((w + 1) * residual - w * residual_un)

            # Standard Diffusion Step
            x_t = scheduler.step(residual, step_t, x_t).prev_sample 
            #compute the norms 
            #mu_norm = x_t.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
            #x_t  = x_t + mu_norm * (scheduler._get_variance(step_t).to(DEVICE) * g_update)
            # Logging
            if step_t % 100 == 0:
                print(f"Step {step_t.item()} | sim={anchor_scale.item():.3f}")
    

    # --- Decode final result ---
    final_z = x_t.clone()
    generated = vae.decode(x_t / vae.config.scaling_factor).sample
    print("the internal classifiaction logits : " , c[0][2])
    # GIF Generation
    if len(masks) > 0:
        masks
        masks_processed = [np.mean(np.abs(m), axis=1, keepdims=True) for m in masks]
        imageio.mimsave("guidance_gradients.gif", 
                        [(m[0, 0] * 255 / (np.max(m) + 1e-8)).astype(np.uint8) for m in masks_processed], 
                        duration=0.1)

    return generated, final_z


def denoise_with_interventions_dvce(
    model, 
    vae, 
    classifier, 
    scheduler, 
    z, 
    intervention_strength=0.05,  # Cc in the paper
    interventions=None,
    distance_strength=0.15,    # Cd in the paper
    noise_start=200            # Paper suggests starting at T/2 (e.g., 200/1000)
):
    DEVICE = z.device
    
    # 1. Initialize State: Add noise to the original latent z
    t_start = torch.tensor([noise_start], device=DEVICE).long()
    noise = torch.randn_like(z)
    x_t = scheduler.add_noise(z, noise, t_start)
    
    # 2. Setup Timesteps (Reverse from noise_start down to 0)
    scheduler.set_timesteps(1000) # Ensure full schedule is available
    timesteps = scheduler.timesteps[scheduler.timesteps <= t_start.item()]

    # 3. Denoising Loop
    for step_t in tqdm(timesteps, desc="DVCE Denoising"):
        
        # --- A. UNet Prediction ---
        with torch.no_grad():
            output = model(x_t, step_t, return_dict=False)
            residual_un = output[0] if isinstance(output, tuple) else output

        # --- B. Gradient Calculation Setup (Equation 12) ---
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            
            # Re-derive x_est (fdn) inside the graph to maintain path to x_in
            alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1, 1, 1, 1)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_ab = torch.sqrt(alpha_bar_t)
            x_est = (x_in - sqrt_1m_ab * residual_un) / sqrt_ab

            # C. Loss Calculations
            # 1. Classifier Guidance
            logits = classifier(x_est)
            total_cls_loss = torch.tensor(0.0, device=DEVICE)
            if interventions:
                for idx, target_val in interventions.items():
                    target_tensor = torch.tensor([target_val], device=DEVICE, dtype=torch.float).expand(logits.size(0))
                    # Note: We use Logits/Log-Probs for grad_proj. 
                    # If using BCE, we negate it to move toward target (Gradient Descent).
                    total_cls_loss += F.binary_cross_entropy_with_logits(logits[:, idx], target_tensor)

            # 2. Distance Guidance (L1 for sparsity)
            dist_loss = F.l1_loss(x_est, z, reduction='sum')

            # D. Compute Gradients w.r.t x_in (xt)
            # Paper uses \nabla_{xt} for both
            grad_cls = torch.autograd.grad(total_cls_loss, x_in, retain_graph=True)[0]
            grad_dist = torch.autograd.grad(dist_loss, x_in)[0]

            # E. Construct g_update (Equation 12)
            # Normalize each component to a unit vector first
            g_cls_unit = unit_normalize(grad_cls)
            g_dist_unit = unit_normalize(grad_dist)
            
            # g_update = Cc * (direction toward target) - Cd * (direction away from original)
            # We subtract grad_cls because it's a loss gradient; we move OPPOSITE the loss.
            g_update = (-intervention_strength * g_cls_unit) - (distance_strength * g_dist_unit)

        # --- F. Adaptive Step Update (Equation 13) ---
        with torch.no_grad():
            # 1. Standard Denoising Step (Get original mu_theta)
            step_output = scheduler.step(residual_un, step_t, x_t)
            mu_theta = step_output.prev_sample
            
            # 2. Extract Variance (Sigma_theta)
            # scheduler._get_variance is a common helper; varies by scheduler class
            variance_t = scheduler._get_variance(step_t).to(DEVICE)
            sigma_t = torch.sqrt(variance_t)
            
            # 3. Apply Adaptive Guidance: mu_t = mu_original + sigma * ||mu|| * g_update
            mu_norm = mu_theta.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
            x_t = mu_theta + (sigma_t * mu_norm * g_update)

    # --- Decode Final Result ---
    z = x_t.clone()
    with torch.no_grad():
        x_t_final = x_t / vae.config.scaling_factor
        generated = vae.decode(x_t_final).sample

    return generated , z 

def load_image(path):
     #loading a specific image
    
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    real_img = transform(image).unsqueeze(0).to(DEVICE)
    attributes = torch.tensor([[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1]], device=DEVICE).float()  # Example attributes
    return real_img, attributes

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
    classifier = load_classifier(config)
    target_attrs = {"Male": 0}
    real_img, attributes = get_sample_with_attributes(IMAGE_DIR, ATTR_PATH, config, target_attrs)


       
    # real_img, attributes = load_sample(config)
    # image_path = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_experiments/IMG_4054.png"
    # real_img, attributes = load_image(image_path)





    if real_img is None:
        print("No matching image found. Exiting.")
        return
    
    #real_img, attributes = load_sample(config)

    z_real, recon_vae = encode_image(vae, real_img)

    interventions = {
         11: -1.0,
         0: -1.0,
    }

    generated_baseline, z_baseline = denoise_with_interventions_dvce(
        model=model,
        vae=vae,
        classifier=classifier,
        scheduler=scheduler,
        z=z_real,
      
        noise_start=200,         # ⬅️ amount of noise added
  
        intervention_strength=0.05,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        distance_strength=0.15
    )

    generated,z = denoise_with_interventions_new(
        model=model,
        vae=vae,
        scheduler=scheduler,
        z=z_real,
        attributes=attributes,
        noise_start=200,         # ⬅️ amount of noise added
        noise_end=202,
        intervention_strength=4,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        w=5,
        classifier=classifier,
    )
    visualize_results(recon_vae, generated_baseline, generated,"intervention_")

   
    # creating a maks for the interventins and then cisualizing the masl 

    #since its a mask we can just compute the sum of all channelsa and then normalize it



    maks_baseline  =( torch.abs(recon_vae - generated_baseline)*2) / torch.max( torch.abs(recon_vae - generated_baseline) )
    maks_cbm  = torch.abs(recon_vae - generated)*2 / torch.max( torch.abs(recon_vae - generated) )


    maks_baseline = torch.mean(maks_baseline, dim=1, keepdim=True)
    maks_cbm = torch.mean(maks_cbm, dim=1, keepdim=True)
    #duplicateteh cahnnels 

    maks_baseline = maks_baseline.repeat(1,3,1,1)
    maks_cbm = maks_cbm.repeat(1,3,1,1)

    maks_baseline = maks_baseline - 1 
    maks_cbm = maks_cbm - 1





    print("maks_baseline ", torch.max(maks_baseline))
    print("maks_cbm ", torch.max(maks_cbm))
    print("mean maks_baseline ", torch.mean(maks_baseline))
    print("mean maks_cbm ", torch.mean(maks_cbm))
    print("mmin maks_baseline ", torch.min(maks_baseline))
    print("min maks_cbm ", torch.min(maks_cbm))
    
    visualize_results(recon_vae, maks_baseline, maks_cbm,text="masks_comparison")
    
    #use the unsharp making to enhance the image 

    #generated_unsharp = recon_vae + 0.5 * (recon_vae - transforms.functional.gaussian_blur(generated, kernel_size=(5,5), sigma=(2.0,2.0)))

    # 1. Isolate the structure of the "No Smile" image
    low_freq_recon = transforms.functional.gaussian_blur(recon_vae, kernel_size=(3, 3), sigma=(11.0,11.0))
    high_freq_recon = recon_vae - low_freq_recon
    # 2. Isolate the details of the "Smile" image
    low_freq_gen = transforms.functional.gaussian_blur(generated, kernel_size=(3,3), sigma=(11.0,11.0))
    high_freq_gen = generated - low_freq_gen
    
    # 3. Combine them
    # This gives you "No Smile" pose with "Smile" noise/texture
    #final_image = low_freq_recon + high_freq_gen
    final_image = low_freq_recon + high_freq_gen

    visualize_results(final_image, low_freq_recon, high_freq_gen,text="freq_swap") 
    visualize_results(recon_vae, generated, final_image,text="cbm_unsharp")

    #compute the nask between the unsharp and the original
    mask_unsharp = torch.abs(recon_vae - final_image)*2 / torch.max( torch.abs(recon_vae - final_image) )
    mask_unsharp = torch.mean(mask_unsharp, dim=1, keepdim=True)
    mask_unsharp = mask_unsharp.repeat(1,3,1,1)
    mask_unsharp = mask_unsharp - 1
    visualize_results(recon_vae, maks_cbm, mask_unsharp,text="cbm_unsharp_mask")
    #encode the final image
    z_final, _ = encode_image(vae, final_image)


    #buildinga spectogram about teh mask 
    plt.figure(figsize=(10,5))
    plt.plot(maks_baseline.flatten().cpu().detach().numpy(), label='Baseline Mask', alpha=0.7)
    plt.plot(maks_cbm.flatten().cpu().detach().numpy(), label='CBM Mask', alpha=0.7)
    plt.title('Mask Comparison')
    plt.xlabel('Pixel Index')
    plt.ylabel('Mask Value')
    plt.legend()
    plt.savefig("mask_comparison_spectrogram.png")

    """
    noisy_z, t = add_noise(scheduler, z,t_low=100, t_high=105)
    
    x_t = denoise(model_noCBM, scheduler, noisy_z.clone(), t)
    x_t = x_t / vae.config.scaling_factor
    generated_noCBM = vae.decode(x_t).sample




  

    x_t = denoise_with_cbm(model, scheduler, noisy_z.clone(), attributes, t)
    z_t = x_t.clone()
    x_t = x_t / vae.config.scaling_factor
    generated = vae.decode(x_t).sample
    

    visualize_results(recon_vae, generated_noCBM, generated,text="no_intervention")

   """
    #checking the classification logits before and after
    classification = classifier(z_real)
    print("Original classification logits: ", classification[:,11])
    classification = classifier(z)
    print("After CBM denoising logits: ", classification[:,11])
    classification = classifier(z_baseline)
    print("After Baseline denoising logits: ", classification[:,11])
    print("#######")
     #checking the classification logits before and after
    classification = classifier(z_real)
    print("Original classification logits: ", classification[:,0])
    classification = classifier(z)
    print("After CBM denoising logits: ", classification[:,0])
    classification = classifier(z_baseline)
    print("After Baseline denoising logits: ", classification[:,0])
    # computung distance between recon_vae, generated_baseline, generated

    l2_recon_baseline = F.mse_loss(recon_vae, generated_baseline).item()
    l2_recon_cbm = F.mse_loss(recon_vae, generated).item()

    print(f"L2 Distance - Recon vs Baseline: {l2_recon_baseline:.4f}")
    print(f"L2 Distance - Recon vs CBM: {l2_recon_cbm:.4f}")

    
    # #intervention  

    # interventions = {
    #     10: 10.0
    # }
    # generated = denoise_with_interventions(
    #     model=model,
    #     vae=vae,
    #     scheduler=scheduler,
    #     z=z,
    #     attributes=attributes,
    #     noise_start=580,         # ⬅️ amount of noise added
    #     noise_end=601,
    #     intervention_strength=1.5,  # ⬅️ controls intensity of interventions
    #     interventions=interventions,
    #     w=10.0
    # )
    
    # visualize_results(real_img, recon_vae, generated,"intervention_1")

    # ############
    # interventions = {
    #     10: 1.0
    # }
    # generated = denoise_with_interventions(
    #     model=model,
    #     vae=vae,
    #     scheduler=scheduler,
    #     z=z,
    #     attributes=attributes,
    #     noise_start=580,         # ⬅️ amount of noise added
    #     noise_end=601,
    #     intervention_strength=1.5,  # ⬅️ controls intensity of interventions
    #     interventions=interventions,
    #     w=10.0
    # )
    # visualize_results(real_img, recon_vae, generated,"intervention_2")
    
    # ############

    # interventions = {
        
    #     10: 5.0
    # }
    # generated = denoise_with_interventions(
    #     model=model,
    #     vae=vae,
    #     scheduler=scheduler,
    #     z=z,
    #     attributes=attributes,
    #     noise_start=580,         # ⬅️ amount of noise added
    #     noise_end=601,
    #     intervention_strength=1.5,  # ⬅️ controls intensity of interventions
    #     interventions=interventions,
    #     w=5.0
    # )
    # visualize_results(real_img, recon_vae, generated,"intervention_3")



if __name__ == "__main__":
    main()
