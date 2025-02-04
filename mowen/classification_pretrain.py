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
from model import MOWEN  # Importing the pre-trained MOWEN model

# **Configuration**
IMG_SIZE = 224
BATCH_SIZE = 128  # Adjusted for multi-GPU
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 5  # Automotive, Environmental, Industrial, Medical, Surveillance
CHECKPOINT_DIR = "checkpoints_classification/"
METRICS_PATH = "metrics/classification_log.csv"
PRETRAINED_WEIGHTS = "checkpoints/mowen_epoch46.pth"  # Update path as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# **Initialize Distributed Training**
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())




import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image

# ? Define Class Mapping Based on File Name Prefix
CLASS_MAPPING = {
    "Surveillance": 0,
    "Medical": 1,
    "Industrial": 2,
    "Environmental": 3,
    "Automotive": 4
}

from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

# ? Custom dataset to extract labels from filenames
class CustomThermalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Define class mapping
        self.class_mapping = {
            "Surveillance": 0,
            "Medical": 1,
            "Industrial": 2,
            "Environmental": 3,
            "Automotive": 4
        }
        
        # ? Scan through all images and extract labels from filenames
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue  # Skip non-folder items
            
            for file in os.listdir(folder_path):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(folder_path, file)
                    label_name = self.extract_label_from_filename(file)
                    
                    if label_name in self.class_mapping:
                        self.image_paths.append(file_path)
                        self.labels.append(self.class_mapping[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def extract_label_from_filename(self, filename):
        # ? Extract label from filename format like "Surveillance_014411_aug01.png"
        for label in self.class_mapping.keys():
            if filename.startswith(label):
                return label
        return None  # If no match, return None


# ? Update DataLoader Function
def get_data_loaders():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = CustomThermalDataset(root_dir="/work/zzv393/mowen-dataset/train", transform=transform)
    val_dataset = CustomThermalDataset(root_dir="/work/zzv393/mowen-dataset/val", transform=transform)

    print(f"? Train dataset size: {len(train_dataset)}")
    print(f"? Validation dataset size: {len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset) if dist.is_initialized() else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=8, pin_memory=True)

    return train_loader, val_loader








# **Classification Head**
class MOWEN_Classifier(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(MOWEN_Classifier, self).__init__()

        self.mowen = MOWEN(img_size=IMG_SIZE, mask_ratio=0.75)

        # **Load Pretrained Weights**
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights, map_location=DEVICE)
            self.mowen.load_state_dict(state_dict, strict=False)
            print("? Loaded MAE-pretrained MOWEN weights!")

        self.mowen.requires_grad_(True)  # Enable training for all layers

        # **New Classification Head**
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)  # Output layer
        )

    def forward(self, x):
        features = self.mowen(x, pretrain=False)  # Extract features from MOWEN
    
        print(f"[DEBUG] Features Shape Before Classification: {features.shape}")  # Debug print
    
        # ? Fix: Apply global pooling if needed
        if features.dim() == 3:  # If (batch, num_patches, 768)
            features = features.mean(dim=1)  # Convert to (batch, 768)
    
        return self.classifier(features)


# **Training Loop**
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if images.shape[1] == 1:  # If grayscale
          images = images.repeat(1, 3, 1, 1)  # Convert to (B, 3, H, W)

        # ? Fix: Ensure labels are `LongTensor`
        labels = labels.long()

        optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")
    return epoch_loss

# **Validation Loop**
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels = labels.long()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(dataloader)
    accuracy = correct / total
    print(f"? Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    return val_loss, accuracy

# **Main Training Function**
def main():
    init_distributed_mode()  # Initialize multi-GPU training
    rank = dist.get_rank()

    print(f"?? Running in Distributed Mode on GPU {rank}")

    train_loader, val_loader = get_data_loaders()

    # ? Debugging Model
    print("? Initializing Classification Model...")
    model = MOWEN_Classifier(pretrained_weights=PRETRAINED_WEIGHTS).to(DEVICE)
    
    # ? Ensure Model Weights Loaded
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"?? NaN detected in parameter: {name}")
    
    
    model = DDP(model, find_unused_parameters=True)

    # **Loss and Optimizer**
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    best_val_acc = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

        # **Save Best Checkpoint**
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mowen_classification_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"? Saved best model at Epoch {epoch+1} with Accuracy: {best_val_acc:.4f}")

        # **Log Metrics**
        if rank == 0:  # Only main process should log
            with open(METRICS_PATH, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_acc])

if __name__ == "__main__":
    main()
