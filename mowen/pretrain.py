import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from model import MOWEN  # Importing your model

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64  # Adjusted for multi-GPU
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_RATIO = 0.75
CHECKPOINT_DIR = "checkpoints/"
METRICS_PATH = "metrics/training_log.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Distributed Training
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

# Dataset and DataLoader
def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = ImageFolder(root="D:/mowen-dataset/train", transform=transform)
    val_dataset = ImageFolder(root="D:/mowen-dataset/val", transform=transform)

    train_sampler = DistributedSampler(train_dataset)  # Distributed sampling
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=8, pin_memory=True)

    return train_loader, val_loader

# Training Loop
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, epoch):
    model.train()
    running_loss = 0.0
    os.makedirs("metrics", exist_ok=True)  # Ensure metrics directory exists

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)):
        images = images.to(DEVICE)

        optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            reconstructed = model(images, pretrain=True)
            cnn_features = model.cnn(images)  # Extract CNN features
            image_patches = model.prepare_patches(cnn_features)  # Convert to patch format
            loss = criterion(reconstructed, image_patches)

        scaler.scale(loss).backward()  # Scaled backprop
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Save sample reconstructions for debugging
        if batch_idx == 0 and dist.get_rank() == 0:  # Save only one batch per epoch
            torch.save(reconstructed[:4], f"metrics/reconstructed_epoch{epoch+1}.pth")

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

            with autocast():
                reconstructed = model(images, pretrain=True)
                cnn_features = model.cnn(images)  # Extract CNN features
                image_patches = model.prepare_patches(cnn_features)  # Convert to patch format
                loss = criterion(reconstructed, image_patches)

            val_loss += loss.item()

    val_loss /= len(dataloader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

# Main Pretraining Function
def main():
    init_distributed_mode()  # Initialize multi-GPU training
    rank = dist.get_rank()

    print(f"Running in Distributed Mode on GPU {rank}")

    train_loader, val_loader = get_data_loaders()

    # Initialize model
    model = MOWEN(img_size=IMG_SIZE, mask_ratio=MASK_RATIO).to(DEVICE)
    model = DDP(model)  # Multi-GPU training

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # Mixed Precision Scaler

    best_val_loss = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion)

        # Save checkpoint every 5 epochs or if best validation loss is achieved
        if epoch % 5 == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mowen_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

        # Log metrics
        if rank == 0:  # Only the main process should log
            with open(METRICS_PATH, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss])

if __name__ == "__main__":
    main()
