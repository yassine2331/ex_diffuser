import os
import csv

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def _find_images(data_dir):
    """Recursively find all image files under data_dir."""
    image_files = []
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                image_files.append(os.path.join(root, fname))
    return sorted(image_files)


class FolderDataSource:
    """Loads image metadata from a CSV file.

    If ``metadata.csv`` does not exist inside *data_dir* it is created
    automatically by scanning the directory for image files, so the caller
    never encounters a ``FileNotFoundError`` when the folder exists but the
    CSV has not been generated yet.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: '{data_dir}'. "
                "Please provide a valid path to the dataset folder."
            )

        metadata_path = os.path.join(data_dir, "metadata.csv")

        if not os.path.exists(metadata_path):
            self._generate_metadata(data_dir, metadata_path)

        self.metadata = self._load_metadata(metadata_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_metadata(self, data_dir, metadata_path):
        """Scan *data_dir* for images and write a ``metadata.csv``."""

        image_files = _find_images(data_dir)

        with open(metadata_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'index'])
            writer.writeheader()
            for idx, img_path in enumerate(image_files):
                rel_path = os.path.relpath(img_path, data_dir)
                writer.writerow({'filename': rel_path, 'index': idx})

    def _load_metadata(self, metadata_path):
        metadata = []
        with open(metadata_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append(row)
        return metadata

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.metadata[idx]


class ShapesDataset(Dataset):
    """PyTorch Dataset that reads images described by a :class:`FolderDataSource`."""

    def __init__(self, source, transform=None):
        self.source = source
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        row = self.source[idx]
        img_path = os.path.join(self.source.data_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'images': image, 'index': int(row['index'])}


class ShapesDataModule:
    """Lightweight data module for folder-based shapes datasets.

    Parameters
    ----------
    data_dir:
        Root directory that contains images (and optionally ``metadata.csv``).
    image_size:
        Height / width to resize images to.
    batch_size:
        Batch size for the returned :class:`~torch.utils.data.DataLoader`.
    num_workers:
        Number of worker processes for data loading.
    """

    def __init__(self, data_dir, image_size=64, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None

    def setup(self):
        full_source = FolderDataSource(self.data_dir)

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.dataset = ShapesDataset(full_source, transform=transform)

    def get_dataloader(self, shuffle=True):
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_dataloader().")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
