import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model.amsam import RefinedAMSAMWithAttentionExtraction
from dataset.SyntheticHyperspectralDataset import SyntheticHyperspectralDataset
import matplotlib.pyplot as plt
from model.amsam import RefinedAMSAM


# Create instances of training and validation datasets
train_dataset = SyntheticHyperspectralDataset(num_samples=100)
val_dataset = SyntheticHyperspectralDataset(num_samples=50)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# 1. Define the RefinedAMSAM variant for attention extraction

class RefinedAMSAMWithAttentionExtraction(RefinedAMSAM):
    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = [att(feature) for att, feature in zip(self.attention, features)]
        
        # Extracting attention maps
        attention_maps = [feature / attended_feature for feature, attended_feature in zip(features, attended_features)]
        
        return self.fusion(x, attended_features, attended_features), attention_maps

# Create an instance of the modified model
model_with_attention_extraction = RefinedAMSAMWithAttentionExtraction(in_channels=100, out_channels=64)
# Assuming you have saved the weights as "model_weights.pth"
# model_with_attention_extraction.load_state_dict(torch.load("model_weights.pth"))

# 2. Function to extract attention maps

def extract_attention_maps(hyperspectral_image, model):
    with torch.no_grad():
        _, attention_maps = model(hyperspectral_image)
    return attention_maps

# 3. Visualize the attention maps

def visualize_attention_maps(attention_maps_list):
    # Loop through attention maps for each scale
    for idx, attention_map in enumerate(attention_maps_list):
        # Compute average attention across all bands
        avg_attention = attention_map.mean(dim=1).squeeze().cpu().numpy()
        
        plt.imshow(avg_attention, cmap='hot')
        plt.colorbar()
        plt.title(f'Attention Map for Scale {idx + 1}')
        plt.show()

# To use:
# Assuming `sample_image` is your hyperspectral image tensor
sample_image = (next(iter(train_loader))[0][0])
print(sample_image.shape)
attention_maps_list = extract_attention_maps(sample_image.unsqueeze(0), model_with_attention_extraction)
visualize_attention_maps(attention_maps_list)