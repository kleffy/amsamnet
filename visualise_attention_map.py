import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from model.amsam import SimpleAMSAMNet, NoAttentionAMSAMNet, TwoScalesAMSAMNet, SingleScaleAMSAMNet
from dataset.hyperspectral_ds_lmdb import HyperspectralPatchLmdbDataset
from loss_functions.kon_losses import NTXentLoss

from model.amsam import RefinedAMSAMWithAttentionExtraction
import matplotlib.pyplot as plt
from model.amsam import RefinedAMSAM


def read_csv_keys(csv_file, csv_file_col_name):
    df = pd.read_csv(csv_file)
    keys = df[csv_file_col_name].tolist()
    return keys
# 1. Define the RefinedAMSAM variant for attention extraction

class RefinedAMSAMWithAttentionExtraction(RefinedAMSAM):
    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = [att(feature) for att, feature in zip(self.attention, features)]
        
        # Extracting attention maps
        attention_maps = [feature / attended_feature for feature, attended_feature in zip(features, attended_features)]
        
        return self.fusion(x, attended_features, attended_features), attention_maps

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
        
        plt.imshow(avg_attention, cmap='nipy_spectral')
        plt.colorbar()
        plt.title(f'Attention Map for Scale {idx + 1}')
        plt.show()


if __name__ == "__main__":
    # Parse the arguments
    if 1:
        config_path = r'/vol/research/RobotFarming/Projects/amsamnet/config/config_test.json'
    else:
        config_path = None
    parser = argparse.ArgumentParser(description='AMSAMNet Training')
    parser.add_argument('-c', '--config', default=config_path,type=str,
                            help='Path to the config file')
    args = parser.parse_args()
    
    config = json.load(open(args.config))

    tag = config["tag"]
    log_dir = config["log_dir"]
    experiment_dir = config["experiment_dir"]
    experiment_name = config["experiment_name"]
    lmdb_save_dir = config["lmdb_save_dir"]
    l1_mean_file = config["l1_mean_file"]
    l1_std_file = config["l1_std_file"]
    l2_mean_file = config["l2_mean_file"]
    l2_std_file = config["l2_std_file"]
    lmdb_file_name = config["lmdb_file_name"]
    columns = config["columns"]
    csv_file_name = config["csv_file_name"]
    in_channels = config["in_channels"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    val_split = config["val_split"]
    out_features = config["out_features"]
    normalize = config["normalize"]
    k = config["k"]
    learning_rate = config["learning_rate"] 
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device.type}')
    # is_gpu = device.type != "cpu"

    keys = read_csv_keys(os.path.join(lmdb_save_dir, csv_file_name), columns[0])

    train_keys, val_keys = train_test_split(keys, test_size=val_split, random_state=42, shuffle=False)

    # Report split sizes
    print("Training set has {} instances".format(len(train_keys)))
    print("Validation set has {} instances".format(len(val_keys)))

    # Instantiate the dataset and the dataloaders
    train_dataset = HyperspectralPatchLmdbDataset(
        train_keys,
        lmdb_save_dir,
        lmdb_file_name,
        in_channels,
        device,
        normalize=normalize,
        mean_file=l1_mean_file,
        std_file=l1_std_file,
    )

    val_dataset = HyperspectralPatchLmdbDataset(
        val_keys,
        lmdb_save_dir,
        lmdb_file_name,
        in_channels,
        device,
        normalize=normalize,
        mean_file=l1_mean_file,
        std_file=l1_std_file,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=0, 
        drop_last=True
    ) 
# To use:
# Assuming `sample_image` is your hyperspectral image tensor
sample_image = (next(iter(train_dataloader))[0][0])
print(sample_image.shape)

# Create an instance of the modified model
model_with_attention_extraction = RefinedAMSAMWithAttentionExtraction(in_channels=224, out_channels=128).to(device)
# Assuming you have saved the weights as "model_weights.pth"
# model_with_attention_extraction.load_state_dict(torch.load("model_weights.pth"))
attention_maps_list = extract_attention_maps(sample_image.unsqueeze(0), model_with_attention_extraction)
visualize_attention_maps(attention_maps_list)