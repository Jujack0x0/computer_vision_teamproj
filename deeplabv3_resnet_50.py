import os
import datetime
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)  # Ensure mask is integer
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to integer class ID (if one-hot encoded)
        if mask.ndim == 3:  # If mask has 3 channels (e.g., one-hot encoded)
            mask = np.argmax(mask, axis=-1)  # Convert to single-channel
        
        # Convert to PyTorch tensors
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask



def get_dataloader(data_dir, batch_size=8):
    # Train Transform (with augmentation)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Validation Transform (without augmentation)
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'train/images'),
        mask_dir=os.path.join(data_dir, 'train/masks'),
        transform=train_transform
    )
    val_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'val/images'),
        mask_dir=os.path.join(data_dir, 'val/masks'),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Create Model
def create_model(num_classes):
    model = fcn_resnet50(pretrained=False)  # Pre-trained backbone
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)  # Modify final layer
    return model

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir, writer):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            # Debugging: Check the shape of masks
            # print(f"Images Shape: {images.shape}, Masks Shape: {masks.shape}")
            
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)  # CrossEntropyLoss expects 3D masks
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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

        # Save model checkpoint
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    # Configurations
    data_dir = "/data/computer_vision_project_seg"  # Change this to your dataset path
    batch_size = 4
    num_classes = 6  # Number of classes (including background)
    num_epochs = 40
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model save directory
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard log directory
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Load data
    train_loader, val_loader = get_dataloader(data_dir, batch_size)

    # Create model
    model = create_model(num_classes)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir, writer)

    # Close TensorBoard writer
    writer.close()
