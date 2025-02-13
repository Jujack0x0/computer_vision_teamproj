import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

mask_path = "/data/computer_vision_project_seg/dataset/class_masked_images/IMG_6326_mask.png"  # 예시 경로
mask = np.array(Image.open(mask_path))
plt.imshow(mask, cmap="gray")
plt.colorbar()
plt.title("Sample Mask Visualization")
plt.show()
print("Unique values in mask:", np.unique(mask))
