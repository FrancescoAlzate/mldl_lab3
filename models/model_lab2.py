import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3),  # 224 -> 112
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),                             # 112 -> 56

            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),                             # 56 -> 28

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),                             # 28 -> 14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)