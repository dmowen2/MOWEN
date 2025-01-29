import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision.utils import save_image
from model import MOWEN

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16  # Adjusted for single-GPU testing
EPOCHS = 5  # Reduced for quick testing
LEARNING_RATE = 1e-4
MASK_RATIO = 0.75
WEIGHTS_PATH = "weights/mowen_pretrained.pth"
METRICS_PATH = "metrics/mowen_metrics.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset and DataLoader
def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = ImageFolder(root="D:/mowen-dataset-test/train", transform=train_transform)
    val_dataset = ImageFolder(root="D:/mowen-dataset-test/val", transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Training Loop
def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)):
        images = images.to(DEVICE)

        # Forward pass with masking for MAE
        reconstructed = model(images, pretrain=True)
        
        # Pass images through CNN first to get feature maps
        cnn_features = model.cnn(images)

        # Now extract patches from CNN features
        image_patches = model.prepare_patches(cnn_features)  # Ensure target is in patch format

        loss = criterion(reconstructed, image_patches)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Save sample reconstructions for debugging
        if batch_idx == 0:  # Save one batch per epoch
            save_image(reconstructed[:4], f"metrics/reconstructed_epoch{epoch+1}.png")
            save_image(images[:4], f"metrics/original_epoch{epoch+1}.png")

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")
    return epoch_loss

# Validation Loop
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)

            # Forward pass
            reconstructed = model(images, pretrain=True)
            # Extract feature patches from images
            with torch.no_grad():
                cnn_features = model.cnn(images)  # Convert full images to CNN feature maps
                image_patches = model.prepare_patches(cnn_features)  # Convert to ViT-compatible patches

            # Compute loss between reconstructed patches and extracted patches
            loss = criterion(reconstructed, image_patches)
            val_loss += loss.item()

    val_loss /= len(dataloader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

# Main Pretraining Function
def main():
    print("Running in single-GPU mode.")

    train_loader, val_loader = get_data_loaders()

    # Initialize model (single GPU)
    model = MOWEN(img_size=IMG_SIZE, mask_ratio=MASK_RATIO).to(DEVICE)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    os.makedirs("weights", exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Model saved with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
