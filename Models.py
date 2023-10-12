"""
CS-7180 Fall 2023
Authors:
1. Jyothi Vishnu Vardhan Kolla
2. Karan Shah

This file contain the code to build the Models used in the Project.
"""

import torch
import torch.nn as nn

# This class implements the CNN model for Illuminant estimation.


class IlluminantEstimationCNN(nn.Module):
    def __init__(self):
        super(IlluminantEstimationCNN, self).__init__()
        # First layer: Conv layer.
        # Input: 32 X 32 X 3.
        # Output: 32 X 32 X 240.
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=240, kernel_size=1, stride=1, padding=0)

        # Max-pooling operation.
        # Input: 32 X 32 X 240.
        # Output: 4 X 4 X 240.
        self.pooling = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        self.relu = nn.ReLU()

        # Fully connected layers.
        # Flatten: 4 X 4 X 240 -> 3840.
        # Linear Layers: 3840 -> 40 -> 4
        self.fc1 = nn.Linear(4 * 4 * 240, 40)
        self.fc2 = nn.Linear(40, 3)

    def forward(self, x):
        x = self.conv1(x)
        # print(f"Shape after first conv:{x.shape}")
        x = self.relu(x)
        x = self.pooling(x)
        # print(f"Shape after maxpool:{x.shape}")
        x = x.view(-1, 4 * 4 * 240)  # Flattening.
        x = self.fc1(x)
        # print(f"Shape after FCN:{x.shape}")
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test():
    model = IlluminantEstimationCNN()
    print(model)

    x = torch.randn(10, 3, 32, 32)
    output = model(x)
    print(f"Final output:{output.size()}")


if __name__ == "__main__":
    test()
