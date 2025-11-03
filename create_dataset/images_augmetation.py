import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

#  Setting the Seed for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# Folder containing original images
original_folder = './monster_image'

# Folder to save augmented images
augmented_folder = './monster_image_augmented'

# Defining Transformations for Data Augmentation
augment_transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(236, scale=(0.95, 1.0)),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.04),
    transforms.ToTensor()
])

# Loading the Original Dataset
original_dataset = datasets.ImageFolder(original_folder)

# Creating the Folder for Augmented Images
os.makedirs(augmented_folder, exist_ok=True)

# Setting the Minimum Number of Images per Class
min_samples = 20

# Looping Through Monster Classes
for class_name in os.listdir(original_folder):
    class_path = os.path.join(original_folder, class_name)  # Path to the class folder
    save_path = os.path.join(augmented_folder, class_name)  # Path to save augmented images
    os.makedirs(save_path, exist_ok=True)

    # Listing Images in the Current Class
    images = [os.path.join(class_path, img) for img in os.listdir(class_path)]

    # Calculating the Number of Images to Generate
    num_original = len(images)  # Number of original images available
    augment_needed = max(0, min_samples - num_original)  # Number of additional images required

    # Saving Original Images
    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")  # Open the image and convert it to RGB format
        img.save(os.path.join(save_path, f"original_{i}.jpg"))  # Save a copy of the original image

    # Generating New Augmented Images
    for i in range(augment_needed):
        img = Image.open(np.random.choice(images)).convert("RGB")  # Select a random image from the class
        img_aug = augment_transform(img)    # Apply augmentation transformations
        img_pil = transforms.ToPILImage()(img_aug)   # Convert back to PIL format for saving
        img_pil.save(os.path.join(save_path, f"augmented_{i}.jpg"))  # Save the augmented image

print("Augmentation completed!")
