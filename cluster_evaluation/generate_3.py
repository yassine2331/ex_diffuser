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
import gc

# === Local imports ===
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.UNet2DWithCBM_new import UNet2DWithCBM
from classifier.celebaClassifer import DiffusersLatentClassifier
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

def save_image(image, path, name):
    #images are 
    image = to_pil(image)
    
    image.save(f"{path}/{name}.png")

def save_images(images, path, start_index):
    #if path do not exist create it 
    if not os.path.exists(path):
        os.makedirs(path)
    names = [f"image_{i}" for i in range(start_index, start_index + len(images))]
    for img, name in zip(images, names):
        save_image(img, path, name)
    return start_index + len(images)



# ==============================================================
# -------------------------- MAIN -------------------------------
# ==============================================================


def get_batch_with_attributes(source_iter, source_loader, config, target_attrs, output_batch_size=16):
    """
    Collects a specific number of images (output_batch_size) that match the target attributes.
    
    Args:
        source_iter (iterator): The persistent iterator (iter(source_loader)).
        source_loader (DataLoader): Needed to reset the iterator if it runs out.
        config: CelebA config.
        target_attrs (dict): Desired attributes, e.g., {"Smiling": 1, "Male": 0}.
        output_batch_size (int): How many matching images you want returned.
    
    Returns:
        (torch.Tensor, torch.Tensor, iterator): (batch_images, batch_attributes, updated_iterator)
    """
    
    # --- Precompute indices of interest ---
    target_indices = {}
    for attr, val in target_attrs.items():
        if attr not in config.selected_attrs:
            raise ValueError(f"'{attr}' not found in selected_attrs.")
        idx = config.selected_attrs.index(attr)
        target_indices[idx] = val

    collected_images = []
    collected_attrs = []
    
    #print(f"🔍 Searching for {output_batch_size} samples with: {target_attrs}")
    checked_count = 0
    collected = 0 
    while collected < output_batch_size:
        try:
            # Grab a chunk of data from the source
            batch = next(source_iter)
        except StopIteration:
            print("↻ End of dataset reached — reshuffling and continuing...")
            source_iter = iter(source_loader)
            batch = next(source_iter)

        images = batch["images"].to(DEVICE)
        attributes = batch["attributes"].float().to(DEVICE)
        checked_count += images.size(0)

        # --- Vectorized Check (Fast) ---
        # 1. Binarize attributes
        attr_bin = (attributes > 0).float()
        
        # 2. Create a mask: Start with all True
        mask = torch.ones(images.size(0), dtype=torch.bool, device=DEVICE)
        
        # 3. Apply conditions for every target attribute
        for idx, target_val in target_indices.items():
            # If we want 1, we check if attr == 1. If we want 0, check if attr == 0
            mask = mask & (attr_bin[:, idx] == target_val)

        # 4. Filter matches
        matches = images[mask]
        match_attrs = attributes[mask]

        if matches.size(0) > 0:
            collected_images.append(matches)
            collected_attrs.append(match_attrs)
            collected += len(matches)

    # --- Final Assembly ---
    # Concatenate all collected chunks
    final_images = torch.cat(collected_images, dim=0)
    final_attrs = torch.cat(collected_attrs, dim=0)

    # Trim to exact batch size (in case we collected too many in the last step)
    final_images = final_images[:output_batch_size]
    final_attrs = final_attrs[:output_batch_size]

    #print(f"✅ Collected {len(final_images)} samples after checking {checked_count} images.")
    
    # Return the iterator too, so you can overwrite the variable outside
    return final_images, final_attrs, source_iter




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

    similarity_weights, ts = [], []

    # print(f"\n⚙️ Starting denoising with interventions at strength {intervention_strength}...")
    # print(f"Noise added at timestep {t.item()}.\n")

    with torch.no_grad():
        for step_t in tqdm(timesteps[tt:]):
            residual_un, c_un = model(x_t, step_t, return_dict=False)

            # --- Apply user-defined concept interventions ---
            if interventions is not None:
                for idx, val in interventions.items():
                    #c_un[:,idx] = val*intervention_strength
                    attributes[:,idx] = val*intervention_strength
            residual, c = model(x_t, step_t, interventions=attributes, return_dict=False)

            # --- Compute cosine similarity for adaptive scaling ---
            alpha_bar_t = scheduler.alphas_cumprod[step_t].to(DEVICE).view(-1, 1, 1, 1)
            alpha_bar_t_prev = scheduler.alphas_cumprod[step_t - 1].to(DEVICE).view(-1, 1, 1, 1) if t > 0 else torch.ones_like(alpha_bar_t)
            beta_t = scheduler.betas[step_t].to(DEVICE).view(-1, 1, 1, 1)
            residual_0 = (x_t - torch.sqrt(alpha_bar_t) * z)/ torch.sqrt(1 - alpha_bar_t)

            similarity = F.cosine_similarity(residual_0.flatten(1), residual.flatten(1), dim=1)
            anchor_scale = (1 + similarity[0]) / 2

            similarity_weights.append(anchor_scale.item())
            ts.append(step_t.item())
            
            
            diffmap = torch.abs(((w + 1) * residual - w * residual_un ) - residual_0)
            mask = diffmap / torch.max(diffmap)
            #bineraize the mask
            threshold = 0.3
            mask = (mask > threshold).float()
            #apply gaussian blur to the mask to smooth it
            mask = transforms.functional.gaussian_blur(mask, kernel_size=(3,3), sigma=(5.0,5.0))

            
            # --- Dynamic blending of residuals (experimental) ---
            if 60 < step_t < 1000:

                residual = mask * ((w + 1) * residual - w * residual_un )+  (1 - mask)* residual_0 # normalization *(torch.norm( (w + 1) * residual - w * residual_un )/(torch.norm(residual_0) ))  
                #residual = (w + 1) * residual  - w * residual_un  +  0.4*residual_0
                #residual =  ((w + 1) * residual - (w + 0.4) * residual_un) +  0.7*(torch.norm( (w + 1) * residual - w * residual_un )/(torch.norm(residual_0) )) * residual_0 
                #residual =  (w + 1) * residual + w *(torch.norm(  residual )/(torch.norm(residual_0) )) * residual_0
                #residual =  (torch.norm(residual_un)/torch.norm(residual))*((w + 1) * residual) + w * residual_0 * (torch.norm( residual_un )/(torch.norm(residual_0) ))
                #residual =  (w + 1) * residual + w * residual_0 * 0.1
                # residual_1 =  ((w + 1) * residual_un + w *(torch.norm(  residual_un )/(torch.norm(residual_0) )) * residual_0)
                # residual_2 =  ((w + 1) * residual + w *(torch.norm(  residual )/(torch.norm(residual_0) )) * residual_0)
                # residual   =  (w + 1)* residual_2 -  w*residual_1
                # residual_2 =  (w-1 ) *(torch.norm(  residual_un )/(torch.norm(residual_0) )) * residual_0
                # residual_1 =  w *(torch.norm(  residual )/(torch.norm(residual_0) )) * residual_0
                # residual   = (w + 1) * residual  - w * residual_un  +  residual_1 - residual_2
                # residual = (torch.norm(residual)/torch.norm(residual_0)) * residual_0 + w * (residual - residual_un) 
            else:
                residual = (w + 1) * residual  - w * residual_un 
                
            x_t = scheduler.step(residual, step_t, x_t).prev_sample

            # if step_t % 100 == 0:
            #     print(f"Step {step_t.item()} | sim={anchor_scale.item():.3f} | c[0][2]={c[0][2].item():.3f}")
            #     #printing the norms of each 
            #     print(f"Norms | residual: {torch.norm(residual).item():.3f}, residual_0: {torch.norm(residual_0).item():.3f}, residual_un: {torch.norm(residual_un).item():.3f}")
            #     print(" -- -- ")
            #     print("norm hmm | ",1/(torch.norm(residual_0) / torch.norm(residual)))

    # --- Decode latent to image ---
    z_t = x_t.clone()
    x_t = x_t / vae.config.scaling_factor
    with torch.no_grad():
        generated = vae.decode(x_t).sample

    # Explicitly delete the latent input copies before returning
    del x_t, noisy_z, noise, t

    return generated , z_t

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
            epsilon_hat = residual_un - sqrt_1m_ab * g_total
            step = scheduler.step(epsilon_hat, step_t, x_t)
            x_t = step.prev_sample

        logits = classifier(x_t)
        print("Final logits after denoising: ", logits)
            
    # --- Decode latent to image ---
    z_t = x_t.clone()
    x_t = x_t / vae.config.scaling_factor
    generated = vae.decode(x_t).sample


    return generated, z_t

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

    # print(f"\n⚙️ Starting denoising with interventions at strength {intervention_strength}...")

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

        # Re-calculate x_est using x_in so the graph is connected
        # x_est = (x_t - sqrt(1-ab) * epsilon) / sqrt(ab)
        x_est = (x_in - sqrt_1m_ab * residual_un.detach()) / sqrt_ab

        # --- C. Classifier Logic ---
        logits = classifier(x_est)
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        if interventions is not None:
            for idx, target_val in interventions.items():
                target_tensor = torch.tensor([target_val]*logits.size(0), device=DEVICE, dtype=torch.float)
                
                # print("target_tensor ", target_tensor)
                # print("logits[:, idx] ", logits[:, idx])
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

        # 2. Gradient of Distance Loss (L1)
        l1_loss = F.l1_loss(x_est, z, reduction='sum')
        # FIX: We take grad w.r.t x_in
        grad_dist = torch.autograd.grad(l1_loss, x_in)[0]

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
        epsilon_hat = residual_un + (sqrt_1m_ab * g_total)
        
        # Step with Scheduler
        x_t = x_t.detach()
        step_output = scheduler.step(epsilon_hat, step_t, x_t)
        x_t = step_output.prev_sample
    
    # Cleanup logits before decoding to save VRAM
    del logits, grad_cls, grad_dist, x_in, x_est

    logits = classifier(x_t)
    # print("Final logits after denoising: ", logits)
    # print("verification :", logits[:,0])
            
    # --- Decode ---
    z_t = x_t.clone()
    x_t = x_t / vae.config.scaling_factor
    with torch.no_grad():
        generated = vae.decode(x_t).sample

    return generated , z_t, logits



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



#defining a function that takesthe hyperparameters and the models and generates counterfactuals 

def CF_generation(model,vae,classifier, scheduler, source_loader, config, target_attrs, interventions , NUMBER_OFITERARIONS=1):
    
    
    start_index_baseline = 0
    start_index_cbm = 0
    start_index_real = 0
    #taking the key name of the target attribut

    folder_name = list(target_attrs.keys())[0]
    print("Generating counterfactuals for target attribute:", folder_name)
    print("the target ", str(target_attrs[folder_name]))
    print("the intervention ", str(int(list(interventions.values())[0])))
    folder_name = folder_name+ "_10" + "/" + folder_name + "_"+ str(target_attrs[folder_name])+"_" + str(int(list(interventions.values())[0]))
    
    source_iter = iter(source_loader)
    for i in tqdm(range(NUMBER_OFITERARIONS)):
        
       
        batch_imgs, attributes, source_iter = get_batch_with_attributes(
        source_iter, 
        source_loader, 
        config, 
        target_attrs, 
        output_batch_size=64
        )

        z, recon_vae = encode_image(vae, batch_imgs)


        generated_baseline , z_baseline, logits_baseline = denoise_with_interventions_baseline(
        model=model,
        vae=vae,
        classifier=classifier,
        scheduler=scheduler,
        z=z,
        attributes=attributes,
        noise_start=200,         # ⬅️ amount of noise added
        noise_end=202,
        intervention_strength=0.1,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        distance_strength=0.15
        )
        
        generated , x_t = denoise_with_interventions(
        model=model,
        vae=vae,
        scheduler=scheduler,
        z=z,
        attributes=attributes,
        noise_start=400,         # ⬅️ amount of noise added
        noise_end=402,
        intervention_strength=1,  # ⬅️ controls intensity of interventions
        interventions=interventions,
        w=10.0
        )
        


        start_index_baseline = save_images(generated_baseline, folder_name + "/images_baseline", start_index_baseline)
        start_index_real = save_images(recon_vae, folder_name + "/images_real", start_index_real)
        start_index_cbm = save_images(generated, folder_name + "/images_cbm", start_index_cbm)

    

def main():

    config = celebA()
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.config.clip_sample = False

    vae = load_vae()
    model = load_ddpm_model(config)
    model_noCBM = load_ddpm_model_noCBM(config)
    classifier = load_classifier(config)
    
    #real_img, attributes = get_sample_with_attributes(IMAGE_DIR, ATTR_PATH, config, target_attrs)

    # a new way to fetch ffor images
    source_loader = get_dataloader(
    IMAGE_DIR,
    ATTR_PATH,
    selected_attrs=config.selected_attrs,
    image_size=128,
    batch_size=64,  # Larger batch size here speeds up scanning
    mode='train',
    num_workers=4   # Increase workers for faster loading
    )
    print("len of source_loader:", len(source_loader))
    #Create a persistent iterator




    target_attrs = {"Smiling":0}
    interventions = {
         2: 1.0
    }

    CF_generation(model,vae,classifier, scheduler, source_loader, config, target_attrs, interventions , 100)


    target_attrs = {"Smiling":1}
    interventions = {
         2: 0.0
    }

    CF_generation(model,vae,classifier, scheduler, source_loader, config, target_attrs, interventions , 100)
        


    target_attrs = {"Young":1}
    interventions = {
         1: 0.0
    }

    CF_generation(model,vae,classifier, scheduler, source_loader, config, target_attrs, interventions , 100)
        

    target_attrs = {"Young":0}
    interventions = {
         1: 1.0
    }

    CF_generation(model,vae,classifier, scheduler, source_loader, config, target_attrs, interventions , 100)


        
    # if real_img is None:
    #     print("No matching image found. Exiting.")
    #     return
    
   
    # #real_img, attributes = load_sample(config)
    # image_path = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/IMG_4054.png"
    # #real_img, attributes = load_image(image_path)





    # visualize_results(recon_vae, generated_baseline, generated,"intervention_")
    # #visualize_results(generated_baseline, generated_baseline, generated_baseline,"intervention_")
    # save_image(recon_vae, "images", "real")
    # save_image(generated_baseline, "images", "baseline")
    # save_image(generated, "images", "cbm")


    #  # priting the prediction of the classifier on each of the images
    # with torch.no_grad():
    #     logits_recon = classifier(z)
    #     logits_baseline = classifier(x_t_baseline)
    #     logits_cbm = classifier(x_t)
    
    # print("logits_recon ", logits_recon[:,2])
    # print("logits_baseline ", logits_baseline[:,2])
    # print("logits_cbm ", logits_cbm[:,2])







if __name__ == "__main__":
    main()
