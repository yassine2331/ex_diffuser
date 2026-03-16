import os
import sys
import torch
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.UNet2DWithCBM_new import UNet2DWithCBM
from classifier.celebaClassifer import DiffusersLatentClassifier
from models.CBM import CBM_new
from configs.diffusion_config import celebA

# ==============================================================
# === CONFIGURATION ===
# ==============================================================
FEATURE_DIM = 2048  
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WHICH ATTRIBUTE INDEX ARE WE CHECKING? (Change this as needed)
# e.g., 2 might be Arched_Eyebrows, 31 might be Smiling. Check your config.
TARGET_ATTR_IDX = 2 

# PATHS & CHECKPOINTS
DDPM_CKPT_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_wandb_299.pt"
CLASSIFIER_CKPT = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/classifier_celeba_final_best.pt"

# Folder Paths
PATH_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_50/Smiling_0_1/images_cbm"      # Generated Images
PATH_BASELINE = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_50/Smiling_0_1/images_baseline" # Baseline/Real Images

# ==============================================================
# ------------------- 1. CUSTOM DATASET -------------------------
# ==============================================================
class FlatFolderDataset(Dataset):
    """
    Reads images from a folder. Returns (C, H, W) tensors.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # Sort by numeric ID to ensure consistent order
        self.files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

# ==============================================================
# ------------------- 2. LOADERS -------------------------------
# ==============================================================
def load_vae():
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    return vae.to(DEVICE).eval()

def load_classifier(config):
    print("Loading CelebA classifier...")
    classifier = DiffusersLatentClassifier(input_channels=16, selected_attributes=config.selected_attrs).to(DEVICE)
    state = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
    classifier.load_state_dict(state)
    return classifier.eval()

def load_ddpm_model(config):
    print("Loading DDPM + CBM model...")
    model = UNet2DWithCBM(config, CBM_new).to(DEVICE)
    state = torch.load(DDPM_CKPT_CBM, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'])
    return model.eval()

def encode_image(vae, img_tensor):
    """Encode image into latent space using VAE."""
    with torch.no_grad():
        latent_dist = vae.encode(img_tensor).latent_dist
        z = latent_dist.sample()
    return z * vae.config.scaling_factor

# ==============================================================
# ------------------------ MAIN ---------------------------------
# ==============================================================
def main():
    print(f"⚡ Device: {DEVICE}")
    print(f"📂 CBM Images (Generated): {PATH_CBM}")
    print(f"📂 Baseline Images (Real): {PATH_BASELINE}")

    # Initialize Config & Models
    config = celebA()
    model = load_ddpm_model(config)
    vae = load_vae()    
    classifier = load_classifier(config)

    # Initialize Datasets
    dataset_cbm = FlatFolderDataset(PATH_CBM)
    dataset_baseline = FlatFolderDataset(PATH_BASELINE)
    
    loader_cbm = DataLoader(dataset_cbm, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    loader_baseline = DataLoader(dataset_baseline, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    
    print(f"📊 Found {len(dataset_cbm)} CBM images and {len(dataset_baseline)} Baseline images.")

    # Storage for results
    results = {
        "CBM": {"classifier_logits": [], "internal_probs": []},
        "Baseline": {"classifier_logits": [], "internal_probs": []}
    }

    # --- Processing Function ---
    def process_dataset(loader, name):
        print(f"\n--- Processing {name} Images ---")
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{name} Batches"):
                batch = batch.to(DEVICE)
                z = encode_image(vae, batch)
                
                # CRITICAL FIX: Pass a tensor of zeros for time step
                # We use t=0 because we are evaluating finished, clean images
                t = torch.zeros(z.shape[0], device=DEVICE, dtype=torch.long)
                
                # 1. Internal CBM Prediction (Returns Probabilities 0-1)
                _, c_un = model(z, t, return_dict=False)
                
                # 2. External Classifier Prediction (Returns Logits -Inf to +Inf)
                class_logits = classifier(z)
                
                results[name]["internal_probs"].append(c_un.cpu().detach().numpy())
                results[name]["classifier_logits"].append(class_logits.cpu().detach().numpy())

    # Run processing
    process_dataset(loader_cbm, "CBM")
    process_dataset(loader_baseline, "Baseline")

    # --- METRICS CALCULATION ---
    print("\n--- Calculating Metrics ---")
    
    # Helper to concatenate and calculate mean
    def get_metrics(name):
        # FIX: Use concatenate, not array, to handle variable batch sizes
        logits = np.concatenate(results[name]["classifier_logits"], axis=0)
        probs = np.concatenate(results[name]["internal_probs"], axis=0)
        
        # Thresholding
        # Classifier is Logits -> > 0.0 is positive
        # Internal is Probs -> >= 0.5 is positive
        pred_classifier = (logits > 0.0).astype(np.float32)
        pred_internal = (probs >= 0.5).astype(np.float32)
        
        # Calculate Mean across all images
        mean_classifier = np.mean(pred_classifier, axis=0)
        mean_internal = np.mean(pred_internal, axis=0)
        
        return mean_classifier, mean_internal

    cbm_mean_ext, cbm_mean_int = get_metrics("CBM")
    base_mean_ext, base_mean_int = get_metrics("Baseline")

    # Output Results
    print(f"\n=== RESULTS FOR ATTRIBUTE INDEX {TARGET_ATTR_IDX} ===")
    print(f"{'Metric':<40} | {'Value':<10}")
    print("-" * 55)
    print(f"{'CBM Images - Classifier (Ext) Accuracy':<40} | {cbm_mean_ext[TARGET_ATTR_IDX]:.4f}")
    print(f"{'CBM Images - Internal Model Accuracy':<40} | {cbm_mean_int[TARGET_ATTR_IDX]:.4f}")
    print("-" * 55)
    print(f"{'Baseline Images - Classifier (Ext) Accuracy':<40} | {base_mean_ext[TARGET_ATTR_IDX]:.4f}")
    print(f"{'Baseline Images - Internal Model Accuracy':<40} | {base_mean_int[TARGET_ATTR_IDX]:.4f}")
    print("=" * 55)

if __name__ == "__main__":
    main()