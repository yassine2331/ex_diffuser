"""Shapes3D dataset wrapper.

The 3D Shapes dataset (https://github.com/google-deepmind/3d-shapes) contains
480 000 64×64 RGB images rendered with six independent generative factors:

    floor_hue    – 10 values in [0, 1)
    wall_hue     – 10 values in [0, 1)
    object_hue   – 10 values in [0, 1)
    scale        –  8 values in [0.75, 1.25]
    shape        –  4 integer values  {0, 1, 2, 3}
    orientation  – 15 values in [-30, 30]

The dataset is distributed as a single HDF5 file (``3dshapes.h5``).  When that
file is not available this module falls back to the folder-based
:class:`~datasets.shapes.ShapesDataModule` so the rest of the pipeline can
still run.
"""

import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Factor metadata ----------------------------------------------------------------
FACTOR_NAMES = [
    'floor_hue',
    'wall_hue',
    'object_hue',
    'scale',
    'shape',
    'orientation',
]

FACTOR_NUM_VALUES = {
    'floor_hue': 10,
    'wall_hue': 10,
    'object_hue': 10,
    'scale': 8,
    'shape': 4,
    'orientation': 15,
}


class Shapes3DDataset(Dataset):
    """Dataset wrapper for the 3D Shapes HDF5 file.

    Parameters
    ----------
    h5_path:
        Path to the ``3dshapes.h5`` file.
    transform:
        Optional torchvision transform applied to each image.  When *None* a
        default ``ToTensor`` + normalisation is used.
    """

    def __init__(self, h5_path, transform=None):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to load the 3D Shapes dataset. "
                "Install it with: pip install h5py"
            ) from exc

        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"3D Shapes HDF5 file not found at '{h5_path}'. "
                "Download it from https://github.com/google-deepmind/3d-shapes"
            )

        self.h5_path = h5_path
        self._file = h5py.File(h5_path, 'r')
        self._images = self._file['images']        # (N, 64, 64, 3) uint8
        self._labels = self._file['labels']        # (N, 6) float64

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        from PIL import Image as PILImage

        img = PILImage.fromarray(self._images[idx])
        image = self.transform(img)

        labels = torch.tensor(self._labels[idx], dtype=torch.float32)

        return {
            'images': image,
            'labels': labels,
        }

    def close(self):
        """Close the underlying HDF5 file handle."""
        if hasattr(self, '_file'):
            try:
                self._file.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
