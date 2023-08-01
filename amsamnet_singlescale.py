import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model.amsam import SingleScaleAMSAMNet

# 1.1. Create a Synthetic Hyperspectral Dataset

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

# Create instances of training and validation datasets
train_dataset = SyntheticHyperspectralDataset(num_samples=100)
val_dataset = SyntheticHyperspectralDataset(num_samples=50)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# Training function
def train(model, dataloader, criterion, optimizer, device="cuda"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    model.to(device)
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Validation function
def validate(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Training loop
num_epochs = 5
print("Training started...")


##################################################################################################
# SINGLE SCALE
##################################################################################################
# 1. Without the Attention Mechanism
from model.amsam import SingleScaleAMSAMNet

# Create an instance of the no attention network
single_scale_model = SingleScaleAMSAMNet(in_channels=100)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(single_scale_model.parameters(), lr=0.001)
# Training loop for the ablation model
# Training loop for the single scale ablation model
train_losses_single_scale, val_losses_single_scale = [], []
train_accuracies_single_scale, val_accuracies_single_scale = [], []

for epoch in range(num_epochs):  # Reducing to 2 epochs for demonstration
    train_loss, train_accuracy = train(single_scale_model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(single_scale_model, val_loader, criterion)
    
    train_losses_single_scale.append(train_loss)
    val_losses_single_scale.append(val_loss)
    train_accuracies_single_scale.append(train_accuracy)
    val_accuracies_single_scale.append(val_accuracy)

    print(f"[Single Scale] Epoch {epoch+1}/2 - Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.2f}%, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2f}%")

print(f"""
      train_losses_single_scale: {train_losses_single_scale}, 
      val_losses_single_scale: {val_losses_single_scale}, 
      train_accuracies_single_scale: {train_accuracies_single_scale}, 
      val_accuracies_single_scale: {val_accuracies_single_scale}
      """)
