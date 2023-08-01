import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from dataset.SyntheticHyperspectralDataset import SyntheticHyperspectralDataset

from model.amsam import TwoScalesAMSAMNet

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
# TWO SCALE
##################################################################################################
# Create an instance of the two scales network
two_scales_model = TwoScalesAMSAMNet(in_channels=100)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(two_scales_model.parameters(), lr=0.001)

# Training loop for the two scales ablation model
train_losses_two_scales, val_losses_two_scales = [], []
train_accuracies_two_scales, val_accuracies_two_scales = [], []

for epoch in range(num_epochs):  # Reducing to 2 epochs for demonstration
    train_loss, train_accuracy = train(two_scales_model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(two_scales_model, val_loader, criterion)
    
    train_losses_two_scales.append(train_loss)
    val_losses_two_scales.append(val_loss)
    train_accuracies_two_scales.append(train_accuracy)
    val_accuracies_two_scales.append(val_accuracy)

    print(f"[Two Scales] Epoch {epoch+1}/2 - Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.2f}%, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2f}%")

print(f"""
      train_losses_two_scales: {train_losses_two_scales}, 
      val_losses_two_scales: {val_losses_two_scales}, 
      train_accuracies_two_scales: {train_accuracies_two_scales}, 
      val_accuracies_two_scales: {val_accuracies_two_scales}
      """)
