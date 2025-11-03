import numpy as np
from torch import nn
from torchvision import transforms, datasets
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from CNN import CNN
import json
import matplotlib.pyplot as plt
import torchvision.utils
import random
from pathlib import Path

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# Define image transformation pipeline for test images
test_transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Custom dataset class for loading test images and extracting class names from filenames
class TestDataset(Dataset):
    def __init__(self, folder_path, class_names, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.class_names = class_names  # List of class names
        self.transform = transform

    def __len__(self):
        return len(self.image_files)  # Return number of images in the folder

    def __getitem__(self, idx):
        # Load image from file
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Extract class name from filename (before the first underscore '_')
        file_name = self.image_files[idx]
        if "_" in file_name:
            true_class_name = file_name.split('_')[0].lower().strip().replace("-", " ")
        else:
            # If no underscore is found, use the whole filename as the class name
            true_class_name = file_name.split('.')[0].lower().strip().replace("-", " ")

            # Check if the extracted class name exists in the predefined class list
        if true_class_name in self.class_names:
            true_class_idx = self.class_names.index(true_class_name)  # Get class index
        else:
            true_class_idx = -1  # Assign -1 for unknown classes

        return image, true_class_idx, self.image_files[idx]  # Return image, class index, and filename


# Function to display an image
def imshow(img):
    img = img / 2 + 0.5  # Denormalize the image
    npimg = img.numpy()  # Convert tensor to NumPy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Rearrange channels for correct display
    plt.axis("off")
    plt.show()


# Define test dataset path
folder_path = '../dataset/monster_image'

# Load test dataset using ImageFolder (organized by class subfolders)
test_data = datasets.ImageFolder(folder_path, transform=test_transform)

# Cross-validation setup
k_folds = 5

# Create DataLoader for test dataset
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define loss function for evaluation
criterion = nn.CrossEntropyLoss()

# Testing phase: Evaluate the best model from each fold
print("\nTesting the best models from each fold:")
test_results = {}
for fold in range(1, k_folds + 1):
    model = CNN().to(device)  # Load CNN model
    model.load_state_dict(torch.load(f"cnn_model_fold_{fold}.pth"))  # Load trained model weights
    model.eval()  # Set model to evaluation mode

    test_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[0]  # Get model predictions
            loss = criterion(output, target)  # Compute loss
            test_loss += loss.item() * data.size(0)  # Accumulate total loss
            _, pred = torch.max(output, 1)  # Get predicted class indices
            correct += (pred == target).sum().item()  # Count correct predictions
            total += target.size(0)  # Update total samples

    # Compute average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    test_results[fold] = {'test_loss': test_loss, 'accuracy': accuracy}
    print(f'Fold {fold}: Test Loss = {test_loss:.6f}, Accuracy = {accuracy:.2f}%')

# Identify the best model based on accuracy
best_fold = max(test_results, key=lambda f: test_results[f]['accuracy'])
print(f"Best model: Fold {best_fold} with Accuracy {test_results[best_fold]['accuracy']:.2f}%")

# Path to folder containing unseen test images
test_path = './test_images'

# Load monster dataset with names and descriptions from JSON file
with open("../Decoder/monsters_embeddings.json", "r") as f:
    monster_data = json.load(f)

# Generate list of class names
class_names = [name.lower() for name in monster_data.keys()]

# Create custom DataLoader for test images
test_data = TestDataset(test_path, class_names, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Test saved models using the test dataset
k_folds = 5
print("\nTesting the best models from each fold:")

for fold in range(1, k_folds + 1):
    model = CNN().to(device)  # Load the CNN model
    model.load_state_dict(torch.load(f"cnn_model_fold_{fold}.pth"))  # Load weights
    model.eval()  # Set to evaluation mode

    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total predictions
    correct_images = []  # List of corrected images

    with torch.no_grad():
        for data, true_class_idxs, filenames in test_loader:
            data = data.to(device)
            output, _ = model(data)  # Get logits
            _, pred_idxs = torch.max(output, 1)  # Get predicted class

            # Loop through each image in the batch
            for i, (fname, true_idx, pred_idx) in enumerate(zip(filenames, true_class_idxs, pred_idxs.cpu().numpy())):

                # Extract the true class name from the index or assign "UNKNOWN" if the index is -1
                true_name = class_names[true_idx] if true_idx != -1 else "UNKNOWN"

                # Extract the predicted class name from the predicted index
                pred_name = class_names[pred_idx]

                # Check if the predicted class matches the true class
                if true_idx == pred_idx:

                    correct += 1  # Increment the correct count
                    print("\n" + "-" * 50)
                    print("\n GUESSED: ", fname)
                    correct_images.append(data[i].cpu())  # Append the correctly classified image to the list

                    # Load the true class image based on the true class name
                    image_path = Path(folder_path) / true_name / f"{true_name}.jpg"
                    true_class_image = Image.open(image_path).convert("RGB")  # Open the true class image in RGB mode

                    transformed_image = test_transform(true_class_image).to(
                        device)  # Apply the test transform and move to device
                    correct_images.append(transformed_image.cpu())  # Append the transformed image to the list

                    # Print the details of the prediction, including the file name, true class, and predicted class
                    print(
                        f"File: {fname.split(".")[0]} | Reale: {true_name} ({true_idx}) | Predetta: {pred_name} ({pred_idx})")
                    print("\n" + "-" * 50 + "\n")

                else:
                    print(
                        f"File: {fname.split(".")[0]} | Reale: {true_name} ({true_idx}) | Predetta: {pred_name} ({pred_idx})")

                total += 1  # Increment the total sample count

    # Compute accuracy for the fold
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nFold {fold} accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}\n")

    # Display correctly classified images
    if correct_images:
        imshow(torchvision.utils.make_grid(correct_images, nrow=2))
    else:
        print("No images classified correctly.")
