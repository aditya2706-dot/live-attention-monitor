import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNN(nn.Module):
    """
    A simple Convolutional Neural Network for attention classification.
    The model is designed to classify an image into one of two states:
    0: Attentive
    1: Distracted
    """
    def __init__(self, num_classes=2):
        super(AttentionCNN, self).__init__()
        # Input: 64x64 grayscale image
        
        # Convolutional Layer 1
        # Input channels: 1, Output channels: 16, Kernel size: 3x3, Padding: 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Max Pooling Layer 1
        # Kernel size: 2x2, Stride: 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 32x32
        
        # Convolutional Layer 2
        # Input channels: 16, Output channels: 32, Kernel size: 3x3, Padding: 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 16x16
        
        # Convolutional Layer 3
        # Input channels: 32, Output channels: 64, Kernel size: 3x3, Padding: 1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max Pooling Layer 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 8x8

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layer 1
        # Input features: 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        
        # Fully Connected Layer 2 (Output Layer)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Raw scores (logits)
        return x