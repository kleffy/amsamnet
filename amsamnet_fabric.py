import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lightning as L

from model.amsam import SimpleAMSAMNet
from dataset.SyntheticHyperspectralDataset import SyntheticHyperspectralDataset

fabric = L.Fabric(accelerator="cuda")
fabric.launch()

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
    # model.to(device)
    for data, labels in dataloader:
        # data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        # loss.backward()
        fabric.backward(loss)
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
    # model.to(device)
    with torch.no_grad():
        for data, labels in dataloader:
            # data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Create an instance of the network
model = SimpleAMSAMNet(in_channels=100)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model, optimizer = fabric.setup(model, optimizer)
train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)

# Training loop
num_epochs = 5
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
print("Training started...")
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"[Full Model] Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.2f}%, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2f}%")

print(f"""
      train_losses: {train_losses}, 
      val_losses: {val_losses}, 
      train_accuracies: {train_accuracies}, 
      val_accuracies: {val_accuracies}
      """)