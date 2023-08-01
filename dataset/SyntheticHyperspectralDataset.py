# 1.1. Create a Synthetic Hyperspectral Dataset

import torch
from torch.utils.data import Dataset


class SyntheticHyperspectralDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=64, num_bands=100):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_bands = num_bands
        self.data = torch.randn(num_samples, num_bands, image_size, image_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]