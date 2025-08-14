
# Convolutional Neural Network (CNN) model definition using PyTorch
import torch
import torch.nn as nn

class VCNN(nn.Module):
    """A CNN-based classifier for Ising model."""
    def __init__(self):
        super(VCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 40, 40)
        x = x.view(-1, 1, 40, 40)  # Add channel dimension
        x = self.cnn(x)
        x = x.view(-1, 4*7*7)
        x = self.classifier(x)
        return x
