import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import MOWEN
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_RATIO = 0.75
WEIGHTS_PATH = "weights/mowen_pretrained.pth"
METRICS_PATH = "metrics/mowen_metrics.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Distributed Training
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    dist.destroy_process_group()

# Dataset and DataLoader
def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = ImageFolder(root="D:/mowen-dataset/train", transform=train_transform)
    val_dataset = ImageFolder(root="D:/mowen-dataset/val", transform=train_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)

    return train_loader, val_loader

# Training Loop
def train_one_epoch(model, dataloader, optimizer, criterion, epoch, rank):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)):
        images = images.to(DEVICE)

        # Forward pass with masking for MAE
        outputs = model(images, pretrain=True)
        loss = criterion(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Save sample reconstructions for debugging
        if batch_idx == 0 and rank == 0:  # Only save once per epoch per rank
            save_image(outputs[:4], f"metrics/reconstructed_epoch{epoch+1}.png")
            save_image(images[:4], f"metrics/original_epoch{epoch+1}.png")

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")
    return epoch_loss

# Validation Loop
def validate_one_epoch(model, dataloader, criterion, rank):
    model.eval()
    val_loss = 0.0
    ssim_total = 0.0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)

            # Forward pass
            outputs = model(images, pretrain=True)
            loss = criterion(outputs, images)
            val_loss += loss.item()

            # Calculate SSIM
            for i in range(images.size(0)):
                orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
                recon_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                ssim_total += ssim(orig_img, recon_img, multichannel=True)

    val_loss /= len(dataloader)
    avg_ssim = ssim_total / len(dataloader.dataset)

    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}, SSIM: {avg_ssim:.4f}")

    return val_loss, avg_ssim

# Log Metrics
def log_metrics(epoch, train_loss, val_loss, avg_ssim):
    if dist.get_rank() == 0:
        os.makedirs("metrics", exist_ok=True)
        metrics_exist = os.path.exists(METRICS_PATH)

        with open(METRICS_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not metrics_exist:
                writer.writerow(["Epoch", "Train Loss", "Validation Loss", "SSIM"])
            writer.writerow([epoch+1, train_loss, val_loss, avg_ssim])

# Main Pretraining Function
def main():
    init_distributed_mode()

    train_loader, val_loader = get_data_loaders()

    # Initialize model and wrap in DDP
    model = MOWEN(img_size=IMG_SIZE, patch_size=16, mask_ratio=MASK_RATIO).to(DEVICE)
    model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    os.makedirs("weights", exist_ok=True)

    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)  # Shuffle for distributed sampler

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, dist.get_rank())
        val_loss, avg_ssim = validate_one_epoch(model, val_loader, criterion, dist.get_rank())

        log_metrics(epoch, train_loss, val_loss, avg_ssim)

        # Save the model if validation loss improves
        if dist.get_rank() == 0 and val_loss < best_val_loss:  # Only rank 0 saves the model
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Model saved with validation loss: {val_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    main()
