import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedKFold
from CNN import CNN
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# folder path of original monsters
folder_path = '../dataset/monster_image'
# folder path of augmented monsters
augmented_folder = '../dataset/monster_image_augmented'

# Create Train and Test sets
train_data = datasets.ImageFolder(augmented_folder, transform=test_transform)
test_data = datasets.ImageFolder(folder_path, transform=test_transform)

# Cross-validation setup
k_folds = 5
# Extract class labels used to balance samples in the generated folds
targets = train_data.targets
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

# Perform Stratified K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_data)), targets)):
    print(f'Fold {fold + 1}/{k_folds}')

    # Create samplers for training and validation data
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create data loaders for training, validation and test sets
    train_loader = DataLoader(train_data, batch_size=50, sampler=train_sampler)
    val_loader = DataLoader(train_data, batch_size=50, sampler=val_sampler)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize the CNN model
    model = CNN().to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=0.001, fused=True)
    n_epochs = 25  # Set the number of training epochs
    best_valid_loss = float('inf')

    # Training and validation loop
    for epoch in range(1, n_epochs + 1):
        model.train()  # Set model to training mode
        train_loss = 0.0

        # Training phase
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Reset gradients
            output = model(data)[0]  # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            train_loss += loss.item() * data.size(0)  # Accumulate total loss

        model.eval()  # Set model to evaluation mode
        valid_loss = 0.0

        # Validation phase (no gradient computation)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)[0]
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        # Compute average training and validation los
        train_loss /= len(train_loader.sampler)
        valid_loss /= len(val_loader.sampler)

        # Print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # Save the model if validation loss improves
        if valid_loss <= best_valid_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.
                  format(best_valid_loss, valid_loss))
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"cnn_model_fold_{fold + 1}.pth")

# Testing phase
print("\nTesting the best models from each fold:")
test_results = {}

# Loop through each saved model from cross-validation
for fold in range(1, k_folds + 1):
    model = CNN().to(device)  # Initialize a new model
    model.load_state_dict(torch.load(f"cnn_model_fold_{fold}.pth"))
    model.eval()  # Set model to evaluation mode

    test_loss = 0.0
    correct = 0
    total = 0

    # Evaluate the model on the test dataset
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # Accumulate total loss
            _, pred = torch.max(output, 1)  # Get predicted class
            correct += (pred == target).sum().item()  # Count correct predictions
            total += target.size(0)  # Count total samples

    # Compute average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total

    # Store test results for each fold
    test_results[fold] = {'test_loss': test_loss, 'accuracy': accuracy}
    print(f'Fold {fold}: Test Loss = {test_loss:.6f}, Accuracy = {accuracy:.2f}%')

# Determine the best model based on accuracy
best_fold = max(test_results, key=lambda f: test_results[f]['accuracy'])
print(f"Best model: Fold {best_fold} with Accuracy {test_results[best_fold]['accuracy']:.2f}%")
