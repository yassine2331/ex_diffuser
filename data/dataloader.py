import torchvision
from datasets import Dataset, Features, Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os


# For concept bottleneck training (return images + concept vectors)
class ConceptMNIST(torch.utils.data.Dataset):
    def __init__(self, data, transform, num_concepts=10):
        self.data = data
        self.transform = transform
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Create one-hot concept vector
        concepts = torch.zeros(self.num_concepts)
        concepts[label] = 1.0  # One concept active per digit
        return {
            "images": self.transform(image),
            "concepts": concepts  # Binary concept vector
        }



# For pure diffusion training (return images only)
class ImageOnlyMNIST(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        return {
            "images": self.transform(image)
        }



def get_dataloader(config, concept=False):
    data_dir = os.path.join(config.root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Load MNIST dataset (train split)
    data = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)

    # Define transform pipeline
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if config.concept:
        

        dataset = ConceptMNIST(data, preprocess, num_concepts=config.num_concepts)

    else:
        
        dataset = ImageOnlyMNIST(data, preprocess)

    return DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)