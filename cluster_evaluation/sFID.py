import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
# Standard FID uses feature=2048. 
# Use 64, 192, or 768 only if comparing to specific non-standard experiments.
FEATURE_DIM = 2048  
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# Paths to your folders
PATH_REAL = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_20/Smiling_0_1/images_real"    # Real CelebA images
#PATH_FAKE = "./fid_generated_images_ddpm2"            # Your generated images
PATH_FAKE = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_20/Smiling_0_1/images_baseline"  # --- IGNORE ---
# --- IGNORE ---
# ==============================================================
# ------------------- 1. CUSTOM DATASET -------------------------
# ==============================================================

class FlatFolderDataset(Dataset):
    """
    Reads images from a folder without needing class subdirectories.
    Returns (C, H, W) uint8 tensors in [0, 255] range.
    """
    # since we are computing the sFID then we need to take the first half if it is real and the second half if it is fake
    def __init__(self, root_dir, real=True):
        self.root_dir = root_dir
        self.files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # files end with a number indicating the id of the image
        self.files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        if real:
            self.files = self.files[:len(self.files)//2]
            
        else:
            self.files = self.files[len(self.files)//2:]
        print(f"Found {len(self.files)} images in {root_dir}")
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        # We transform to Tensor (uint8) immediately
        # torchmetrics expects uint8 [0, 255] or float [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)), # Resize to Inception input size
            transforms.PILToTensor(),       # Keeps it uint8 [0-255] (Saves Memory)

        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

# ==============================================================
# ------------------------ MAIN ---------------------------------
# ==============================================================

def main():
    print(f"⚡ Device: {DEVICE}")
    print(f"📂 Real Images: {PATH_REAL}")
    print(f"📂 Fake Images: {PATH_FAKE}")
    
    # 1. Initialize TorchMetrics FID
    # feature=2048 is the standard for academic papers. 
    fid = FrechetInceptionDistance(feature=FEATURE_DIM,normalize=True).to(DEVICE)
    
    # 2. Setup DataLoaders
    # We use num_workers to load images in parallel while GPU processes previous batch
    dataset_real = FlatFolderDataset(PATH_REAL, real=False)
    dataset_fake = FlatFolderDataset(PATH_FAKE, real=False)
    
    loader_real = DataLoader(dataset_real, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    loader_fake = DataLoader(dataset_fake, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    print(f"📊 Found {len(dataset_real)} real images and {len(dataset_fake)} fake images.")

    i = 0
    # 4. Process Fake Images
    print("\n--- Processing Fake Images ---")
    for batch in tqdm(loader_fake, desc="Fake Batches"):
        batch = batch.to(DEVICE)
        fid.update(batch, real=False)
        i += 1

    

    j = 0 
    # 3. Process Real Images
    print("\n--- Processing Real Images ---")
    for batch in tqdm(loader_real, desc="Real Batches"):
        batch = batch.to(DEVICE)
        fid.update(batch, real=True)
        if i *3== j:
            break  # Ensure equal number of batches
        j += 1

    

    # 5. Compute Final Score
    print("\n🧮 Computing final FID score...")
    result = fid.compute()
    
    print("==================================================")
    print(f"🏆 FID Score: {result.item():.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()