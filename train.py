"""
CS-7180 Fall 2023
Authors:
1. Jyothi Vishnu Vardhan Kolla
2. Karan Shah

This file contain the code to train the model.
"""
import configparser
from utils import load_data
import torch.nn.functional as F
import torch.optim as optim
from Models import IlluminantEstimationCNN


import torch

def euclidean_distance_loss(y_true, y_pred):
    """
    Compute the Euclidean distance between two tensors.

    Parameters:
    - y_true (torch.Tensor): the ground truth tensor
    - y_pred (torch.Tensor): the predicted tensor
    
    Returns:
    - torch.Tensor: the Euclidean distance between y_true and y_pred
    """
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1)).mean()



def main(shigehler_config):
    patches, patch_labels, image_idx = load_data("data/shigehler.npz")
    print(f"{patches.shape} {patch_labels.shape} {image_idx.shape}")
    model = IlluminantEstimationCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=0.0005)

if __name__ == "__main__":
    shigehler_config = configparser.ConfigParser()
    shigehler_config.read("assests/shigehler.cfg")

    main(shigehler_config)
