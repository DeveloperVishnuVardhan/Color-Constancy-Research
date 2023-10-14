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
from torch.utils.data import DataLoader, TensorDataset


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

    # Convert numpy arrays to torch tensors.
    patches_tensors = torch.from_numpy(patches).permute(0, 3, 1, 2).float()
    patches_labels_tensor = torch.from_numpy(patch_labels).float()

    # Split the dataset into train and val.
    num_images = 568
    patches_per_image = 100
    train_images = int(0.8 * num_images)
    train_size = train_images * patches_per_image

    train_patches, train_patch_labels, train_idx = patches_tensors[
        :train_size], patches_labels_tensor[:train_size], image_idx[:train_size]
    val_patches, val_patch_labels, val_idx = patches_tensors[
        train_size:], patches_labels_tensor[train_size:], image_idx[train_size:]

    print(f"{train_patches.shape}, {train_patch_labels.shape}, {train_idx.shape}")
    print(f"{val_patches.shape}, {val_patch_labels.shape}, {val_idx.shape}")

    train_dataset = TensorDataset(train_patches, train_patch_labels)
    val_dataset = TensorDataset(val_patches, val_patch_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"{train_loader}, {val_loader}")

    device = "mps"
    model = IlluminantEstimationCNN().to(device)
    # Training loop with validation.
    num_epochs = 20
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = euclidean_distance_loss(target, output)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = euclidean_distance_loss(target, output)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Print statistics
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        # Save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training Finished!")


if __name__ == "__main__":
    shigehler_config = configparser.ConfigParser()
    shigehler_config.read("assests/shigehler.cfg")

    main(shigehler_config)
