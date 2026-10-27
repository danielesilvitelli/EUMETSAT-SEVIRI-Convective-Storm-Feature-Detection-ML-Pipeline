"""
Training Module for EUMETSAT SEVIRI Convective Storm Feature Detection.
Handles model training, optimization, and meteorological performance evaluation (CSI).
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.unet_model import SimpleUNet
from src.data_pipeline import run_data_pipeline


def calculate_csi(preds, targets, threshold=0.5):
    """
    Calculate Critical Success Index (CSI) for binary convective storm masks.
    CSI = Hits / (Hits + Misses + False Alarms)
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()
    
    intersection = (preds_bin * targets_bin).sum()
    total_union = preds_bin.sum() + targets_bin.sum() - intersection
    
    if total_union == 0:
        return torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
        
    return (intersection / total_union).item()


def train_model(dataloader, model, device, epochs=10, lr=0.001):
    """Execute the training loop using Binary Cross Entropy Loss and Adam optimizer."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    print("Starting training for Convective Storm Feature Detection...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_csi = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            # Generate physical ground truth target mask based on cloud-top temperature threshold
            target_mask = (batch[:, 0:1, :, :] < 0.3).float()
            
            optimizer.zero_grad()
            output = model(batch)
            
            loss = criterion(output, target_mask)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_csi += calculate_csi(output, target_mask)
            
        epoch_loss = running_loss / len(dataloader)
        epoch_csi = running_csi / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Meteorological CSI Score: {epoch_csi:.4f}")
        
    return model


if __name__ == "__main__":
    # Set up device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define file paths
    RAW_FILE = "data/raw/MSG3-SEVI-MSG15-0100-NA-20261024141243.420000000Z-NA.nat"
    PROCESSED_IMG = "data/processed/europe_ir108.png"
    
    # 1. Ingest data and initialize PyTorch DataLoader via the data pipeline module
    dataloader = run_data_pipeline(RAW_FILE, PROCESSED_IMG)
    
    # 2. Initialize the U-Net model and move it to the execution device
    model = SimpleUNet(in_channels=2, out_channels=1).to(device)
    
    # 3. Execute the training pipeline
    trained_model = train_model(dataloader, model, device, epochs=10, lr=0.001)
    
    # Save trained weights for inference or deployment
    torch.save(trained_model.state_dict(), "data/processed/unet_seviri_weights.pth")
    print("Training complete! Model weights successfully saved.")