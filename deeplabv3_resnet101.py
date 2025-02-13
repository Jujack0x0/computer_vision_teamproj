import os
import datetime
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet101
from albumentations import Compose, Resize, Normalize, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=6):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)  # Ensure mask is integer

        # Ensure mask values are within valid range
        mask[mask >= self.num_classes] = 0  # Assign invalid values to background (class 0)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to PyTorch tensors
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask


# Function to create the model
def create_model(num_classes):
    model = deeplabv3_resnet101(pretrained=False)  # More powerful model
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Modify final layer
    return model

# Function to calculate class weights
def calculate_class_weights(train_loader, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for _, masks in train_loader:
        for mask in masks:
            unique, counts = np.unique(mask.numpy(), return_counts=True)
            class_counts[unique] += counts
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)

# Training Function
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir, writer, class_weights):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Add class weights
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        # TensorBoard logs
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Validation", val_loss / len(val_loader), epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        # Save model checkpoint
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    # Configurations
    data_dir = "/data/computer_vision_project_seg"  # Change to your dataset path
    batch_size = 4
    num_classes = 6  # Number of classes (including background)
    num_epochs = 40
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard log directory
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Model save directory
    save_dir = "/data/computer_vision_project_seg/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Define data augmentations
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(p=0.3),
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Load datasets
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'train/images'),
        mask_dir=os.path.join(data_dir, 'train/masks'),
        transform=train_transform,
        num_classes=num_classes
    )
    val_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'val/images'),
        mask_dir=os.path.join(data_dir, 'val/masks'),
        transform=val_transform,
        num_classes=num_classes
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Check unique values in masks
    for _, masks in train_loader:
        unique_values = np.unique(masks.numpy())
        print("Unique values in masks:", unique_values)
        break
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Debugging: Check unique values in masks
    for _, masks in train_loader:
        unique_values = np.unique(masks.numpy())
        print("Unique values in masks:", unique_values)
        break

    # Calculate class weights
    class_weights = calculate_class_weights(train_loader, num_classes)
    print("Class weights:", class_weights)

    # Create model
    model = create_model(num_classes)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir, writer, class_weights)

    # Close TensorBoard writer
    writer.close()
