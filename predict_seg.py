import os
import torch
import numpy as np
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# Function to create the model
def create_model(num_classes):
    model = fcn_resnet50(pretrained=False)  # Pre-trained backbone
    model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1)  # Modify final layer
    return model

# Function to load test dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        original_image = image.copy()  # Save original image for visualization

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, original_image, self.images[idx]

# Function to save predictions as images and print class names
def save_predictions_with_classes(model, test_loader, device, save_dir, class_colors, class_mapping):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for images, original_images, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get class predictions

            for pred, original_image, image_name in zip(preds, original_images, image_names):
                # Resize segmentation map to match the original image size
                pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(original_image.shape[:2][::-1], resample=Image.NEAREST))

                # Create a colored segmentation map
                segmentation_map = np.zeros((*pred_resized.shape, 3), dtype=np.uint8)
                for class_id, color in class_colors.items():
                    segmentation_map[pred_resized == class_id] = color

                # Ensure both original_image and segmentation_map are NumPy arrays
                original_image = np.array(original_image)  # Convert PIL image or tensor to NumPy array
                overlay = (0.5 * original_image + 0.5 * segmentation_map).astype(np.uint8)  # Create overlay

                # Save the result
                save_path = os.path.join(save_dir, f"result_{image_name}")
                overlay_image = Image.fromarray(overlay)
                overlay_image.save(save_path)

                # Extract unique class IDs from the prediction
                unique_classes = np.unique(pred_resized)

                # Print class names based on the mapping
                class_names = [class_mapping[class_id] for class_id in unique_classes]
                print(f"Image: {image_name} contains classes: {', '.join(class_names)}")



# Main function for testing
if __name__ == "__main__":
    # Configurations
    data_dir = "/data/computer_vision_project_seg/test/images"  # Path to test images
    model_path = "/data/computer_vision_project_seg/model_epoch_40.pth"  # Path to the saved model
    save_dir = "/data/computer_vision_project_seg/test_results"  # Directory to save test results
    num_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define class colors for visualization (example: RGB colors for each class)
    class_colors = {
        0: [0, 0, 0],       # Background - Black
        1: [255, 0, 0],     # Class 1 - Red
        2: [0, 255, 0],     # Class 2 - Green
        3: [0, 0, 255],     # Class 3 - Blue
        4: [255, 255, 0],   # Class 4 - Yellow
        5: [255, 0, 255],   # Class 5 - Magenta
    }

    # Define class mapping for class names
    class_mapping = {
        0: "background",
        1: "rice",
        2: "salad",
        3: "popcorn chicken",
        4: "danmuji",
        5: "donggeurangddaeng"
    }

    # Test transform
    test_transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Load test dataset
    test_dataset = TestDataset(image_dir=data_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Save predictions and print class names
    save_predictions_with_classes(model, test_loader, device, save_dir, class_colors, class_mapping)

    print(f"Test results saved to {save_dir}")
