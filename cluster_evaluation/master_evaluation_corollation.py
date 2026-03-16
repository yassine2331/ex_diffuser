import os
# --- ADDED: FORCE TENSORFLOW TO CPU TO AVOID CUDNN VERSION CRASH ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# ------------------------------------------------------------------

import sys
import torch
import gc # Added for memory clearing
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import AutoencoderKL
from deepface import DeepFace

# Ensure parent directory is in path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.UNet2DWithCBM_new import UNet2DWithCBM
from classifier.celebaClassifer import DiffusersLatentClassifier
from models.CBM import CBM_new
from configs.diffusion_config import celebA

# ==============================================================
# 🎛️ MASTER CONFIGURATION
# ==============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_WORKERS = 4
DEEPFACE_SAMPLES = 1000 

EXPERIMENTS = [
         {
        "task_name": "Corrolation",
        "direction": "0 -> 1",
        "target_attr_idx": [0, 11],  
        "target_label": 0,
        "source_folder": "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Corrolation_final_2/images_real",
        "models": {
            "MyModel_CBM_final_5": "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Corrolation_final_2/images_cbm",
            "Baseline_final_2":    "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Corrolation_final_2/images_baseline"
        }
    },
       

      

]

DDPM_CKPT_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/concept_model_wandb_299.pt"
CLASSIFIER_CKPT = "/home/oueslatiy/MasterThesis/ex_diffuser/experiments/CelebA/classifier_celeba_final_best.pt"

# ==============================================================
# 🛠️ DATASET UTILITIES
# ==============================================================

class GenericDataset(Dataset):
    def __init__(self, root_dir, transform=None, sfid_mode=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
        
        if sfid_mode == "real": 
            self.files = self.files[len(self.files)//2:]
        elif sfid_mode == "fake": 
            self.files = self.files[:len(self.files)//2]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img) if self.transform else img

# ==============================================================
# 📏 METRIC IMPLEMENTATIONS
# ==============================================================

def calculate_flip_rates(loader, vae, model, classifier, target_idxs, target_label):
    ext_logits1, int_probs1 = [], []
    ext_logits2, int_probs2 = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="   Flip Rate Analysis", leave=False):
            batch = batch.to(DEVICE)
            latent_dist = vae.encode(batch).latent_dist
            z = latent_dist.sample() * vae.config.scaling_factor
            t = torch.zeros(z.shape[0], device=DEVICE, dtype=torch.long)
            _, c_un = model(z, t, return_dict=False)
            class_logits = classifier(z)
            ext_logits1.append(class_logits[:, target_idxs[0]].cpu().numpy())
            int_probs1.append(c_un[:, target_idxs[0]].cpu().numpy())
            ext_logits2.append(class_logits[:, target_idxs[1]].cpu().numpy())
            int_probs2.append(c_un[:, target_idxs[1]].cpu().numpy())
            

    ext_logits1 = np.concatenate(ext_logits1)
    int_probs1 = np.concatenate(int_probs1)
    ext_logits2 = np.concatenate(ext_logits2)
    int_probs2 = np.concatenate(int_probs2)
    

    if target_label == 1:
        ext_success = ((ext_logits1 > 0.0) & (ext_logits2 > 0.0)).mean()
        int_success = ((int_probs1 >= 0.5) & (int_probs2 >= 0.5)).mean()
    else:
        ext_success = ((ext_logits1 < 0.0) & (ext_logits2 < 0.0)).mean()
        int_success = ((int_probs1 < 0.5) & (int_probs2 < 0.5)).mean()
    return ext_success, int_success

def calculate_proximity(source_folder, gen_folder):
    """
    Computes L1, L2, and L1.5 using PyTorch batches for maximum speed.
    """
    # 1. Setup specific Loader for Proximity (No normalization, just 0-1 tensors)
    tf_prox = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # We use our existing GenericDataset to ensure filenames match image_0, image_1...
    ds_src = GenericDataset(source_folder, transform=tf_prox)
    ds_gen = GenericDataset(gen_folder, transform=tf_prox)
    
    # Use a large batch size; pixel math is cheap on VRAM
    dl_src = DataLoader(ds_src, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_gen = DataLoader(ds_gen, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    l1_total, l2_total, l15_total = 0.0, 0.0, 0.0
    total_images = 0

    print("   👉 Phase: Batch Pixel Proximity")
    with torch.no_grad():
        # Zip the loaders to compare Source[i] with Gen[i]
        for batch_s, batch_g in tqdm(zip(dl_src, dl_gen), total=len(dl_src), desc="   Pixel Dist", leave=False):
            batch_s = batch_s.to(DEVICE)
            batch_g = batch_g.to(DEVICE)
            
            # Compute absolute difference
            diff = torch.abs(batch_s - batch_g)
            
            # L1: Mean across channels and pixels, then sum for the batch
            l1_total += torch.mean(diff, dim=(1, 2, 3)).sum().item()
            
            # L2: Root Mean Square
            # (Mean of diff^2)^(1/2) per image
            l2_per_img = torch.sqrt(torch.mean(diff**2, dim=(1, 2, 3)))
            l2_total += l2_per_img.sum().item()
            
            # L1.5: (Mean of diff^1.5)^(1/1.5) per image
            l15_per_img = torch.pow(torch.mean(torch.pow(diff + 1e-8, 1.5), dim=(1, 2, 3)), 1/1.5)
            l15_total += l15_per_img.sum().item()
            
            total_images += batch_s.size(0)

    # Return averages across the entire dataset
    return (l1_total / total_images), (l2_total / total_images), (l15_total / total_images)


def calculate_deepface(source_folder, gen_folder, samples):
    import logging
    logging.getLogger("deepface").setLevel(logging.ERROR)
    
    src_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.png')])
    gen_files = sorted([f for f in os.listdir(gen_folder) if f.endswith('.png')])
    limit = min(len(src_files), len(gen_files), samples)
    verified, dists = 0, []
    
    for i in tqdm(range(limit), desc="   Face Identity", leave=False):
        try:
            res = DeepFace.verify(
                os.path.join(source_folder, src_files[i]), 
                os.path.join(gen_folder, gen_files[i]), 
                distance_metric="cosine", 
                enforce_detection=False,
                silent=True 
            )
            dists.append(res["distance"])
            if res["verified"]: verified += 1
        except Exception: 
            continue
            
    return (verified / limit) * 100, np.mean(dists) if dists else 0

def calculate_sfid(real_folder, fake_folder):
    tf_sfid = transforms.Compose([transforms.Resize((299, 299)), transforms.PILToTensor()])
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    
    ds_real = GenericDataset(real_folder, transform=tf_sfid, sfid_mode="fake")
    ds_fake = GenericDataset(fake_folder, transform=tf_sfid, sfid_mode="real")
    
    dl_real = DataLoader(ds_real, batch_size=BATCH_SIZE, shuffle=False)
    dl_fake = DataLoader(ds_fake, batch_size=BATCH_SIZE, shuffle=False)

    for batch in dl_real: fid_metric.update(batch.to(DEVICE), real=True)
    for batch in dl_fake: fid_metric.update(batch.to(DEVICE), real=False)
    
    return fid_metric.compute().item()

# ==============================================================
# 🚀 MAIN LOOP
# ==============================================================

def main():
    config = celebA()
    # Loading Shared Models
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae").to(DEVICE).eval()
    classifier = DiffusersLatentClassifier(input_channels=16, selected_attributes=config.selected_attrs).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE))
    classifier.eval()
    
    model = UNet2DWithCBM(config, CBM_new).to(DEVICE)
    model.load_state_dict(torch.load(DDPM_CKPT_CBM, map_location=DEVICE)['model_state_dict'])
    model.eval()

    tf_vae = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    results_list = []

    for exp in EXPERIMENTS:
        print(f"\nEvaluating: {exp['task_name']} | {exp['direction']}")
        
        for m_name, m_path in exp["models"].items():
            print(f" > Model: {m_name}")
            
            # 1. Flip Rates
            ds_flip = GenericDataset(m_path, transform=tf_vae)
            ld_flip = DataLoader(ds_flip, batch_size=BATCH_SIZE, shuffle=False)
            ext_f, int_f = calculate_flip_rates(ld_flip, vae, model, classifier, exp["target_attr_idx"], exp["target_label"])
            
            # 2. Pixel Proximity
            l1, l2, l15 = calculate_proximity(exp["source_folder"], m_path)
            
            # 3. Face ID (DeepFace) - Safely runs on CPU due to top env var
            #id_p, cos_s = calculate_deepface(exp["source_folder"], m_path, DEEPFACE_SAMPLES)
            id_p, cos_s = 0,0
            
            
            # 4. sFID (Distribution Plausibility)
            sfid = calculate_sfid(exp["source_folder"], m_path)

            results_list.append({
                "Task": exp["task_name"], "Dir": exp["direction"], "Model": m_name,
                "Flip (Ext)": round(ext_f, 4), "Flip (Int)": round(int_f, 4),
                "sFID": round(sfid, 2), "Face ID %": round(id_p, 2), "Face Cosine": round(cos_s, 4),
                "L1": round(l1, 4), "L2": round(l2, 4), "L1.5": round(l15, 4)
            })

    # Clear GPU after the heavy lifting
    del vae, classifier, model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(results_list)
    df.to_csv("master_evaluation_CORR.csv", index=False)
    print("\nFinal Results:\n", df.to_string(index=False))

if __name__ == "__main__":
    main()