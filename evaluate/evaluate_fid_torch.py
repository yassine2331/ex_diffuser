import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# Custom Dataset for folder-based images
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # → float32 in [0,1]
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # → uint8 in [0,255]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        return self.transform(img)

# Function to compute FID
def compute_fid_from_folders(real_folder, fake_folder, batch_size=64, device="cuda"):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    real_dataset = ImageFolderDataset(real_folder)
    fake_dataset = ImageFolderDataset(fake_folder)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    print(f"→ Found {len(real_dataset)} real images and {len(fake_dataset)} fake images")
    print("→ Extracting features and computing FID...")

    # Real images
    for batch in tqdm(real_loader, desc="Processing real images"):
        fid.update(batch.to(device), real=True)

    # Fake images
    for batch in tqdm(fake_loader, desc="Processing fake images"):
        fid.update(batch.to(device), real=False)

    # Compute final score
    score = fid.compute().item()
    print(f"\n✅ FID Score (TorchMetrics): {score:.4f}")
    return score

# Main block
if __name__ == "__main__":
    real_dir = "./real_mnist"
    fake_dir = "./fake_mnist"
    compute_fid_from_folders(real_dir, fake_dir, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu")
