import torchvision
from datasets import Dataset, Features, Image
from torchvision import transforms
import torch
import os

def get_dataloader(config):
    data_dir = os.path.join(config.root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    data = torchvision.datasets.MNIST(root=data_dir, download=True)

    features = Features({"image": Image()})
    dataset = Dataset.from_dict({"image": data.data}, features=features)
    dataset = dataset.with_format("torch")

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform(examples):
        images = [preprocess(image) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    return torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
