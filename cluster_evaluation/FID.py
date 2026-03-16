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
PATH_REAL = "/home/oueslatiy/data/celeba/images"    # Real CelebA images
#PATH_FAKE = "./fid_generated_images_ddpm2"            # Your generated images
PATH_FAKE = "./fid_generated_images_CBMddpm"  # --- IGNORE ---
# ==============================================================
# ------------------- 1. CUSTOM DATASET -------------------------
# ==============================================================

class FlatFolderDataset(Dataset):
    """
    Reads images from a folder without needing class subdirectories.
    Returns (C, H, W) uint8 tensors in [0, 255] range.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        # We transform to Tensor (uint8) immediately
        # torchmetrics expects uint8 [0, 255] or float [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)), # Resize to Inception input size
            transforms.PILToTensor()       # Keeps it uint8 [0-255] (Saves Memory)
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
    fid = FrechetInceptionDistance(feature=FEATURE_DIM).to(DEVICE)
    
    # 2. Setup DataLoaders
    # We use num_workers to load images in parallel while GPU processes previous batch
    dataset_real = FlatFolderDataset(PATH_REAL)
    dataset_fake = FlatFolderDataset(PATH_FAKE)
    
    loader_real = DataLoader(dataset_real, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    loader_fake = DataLoader(dataset_fake, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

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