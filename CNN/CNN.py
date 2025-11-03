from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer: input (3 channels), output (16 channels), kernel size 3x3, padding 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization to stabilize training

        # Second convolutional layer: input (16 channels), output (32 channels), kernel size 3x3, padding 1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization to stabilize training

        # Third convolutional layer: input (32 channels), output (64 channels), kernel size 3x3, padding 1
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization to stabilize training

        # Max pooling layer: reduces spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)

        # First fully connected layer: input features (64 * 29 * 29), output 8192 neurons
        self.fc1 = nn.Linear(64 * 29 * 29, 8192)
        self.bn_fc1 = nn.BatchNorm1d(8192)  # Batch normalization to stabilize training

        # Second fully connected layer: input 8192 neurons, output 4096 neurons
        self.fc2 = nn.Linear(8192, 4096)

        # Third fully connected layer: input 4096 neurons, output 2048 neurons
        self.fc3 = nn.Linear(4096, 2048)

        # Fourth fully connected layer: input 2048 neurons, output 768 neurons
        self.fc4 = nn.Linear(2048, 768)

        # Fifth fully connected layer: input 768 neurons, output 222 neurons
        self.fc5 = nn.Linear(768, 222)

        # Dropout layer to prevent overfitting (drop probability 0.50)
        self.dropout = nn.Dropout(0.65)

    def forward(self, x):
        # Convolutional layers with BatchNorm, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor to feed into fully connected layers
        x = x.view(x.size(0), -1)

        # Fully Connected layers with GELU activation and Dropout
        x = self.dropout(F.gelu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.gelu(self.fc2(x)))
        x = self.dropout(F.gelu(self.fc3(x)))
        x = self.dropout(self.fc4(x))

        # Normalize the 768-dimensional embedding vector
        embedding = F.normalize(x, p=2, dim=1)

        # Final classification layer
        x = self.fc5(embedding)

        return x, embedding
