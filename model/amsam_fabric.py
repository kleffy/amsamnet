
from typing import Any
import torch
import lightning as L
from torch import nn


from model.amsam import SimpleAMSAMNet

class AmsamFabric(L.LightningModule):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.model = SimpleAMSAMNet(in_channels=in_channels, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def training_step(self, batch, batch_idx):
        data, labels = batch
        
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        return loss
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)