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
from scipy.io import loadmat
import numpy as np
import torch
import pandas as pd
import os
import subprocess


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

def compute_angular_error(y_true, y_pred):
    """
    Angle between the RGB triplet of the measured ground truth
    illumination and RGB triplet of estimated illuminant

    Args:
        y_true (np.array): ground truth RGB illuminants
        y_pred (np.array): predicted RGB illuminants
    Returns:
        err (np.array):  angular error
    """

    gt_norm = np.linalg.norm(y_true, axis=1)
    gt_normalized = y_true / gt_norm[..., np.newaxis]
    est_norm = np.linalg.norm(y_pred, axis=1)
    est_normalized = y_pred / est_norm[..., np.newaxis]
    dot = np.sum(gt_normalized * est_normalized, axis=1)
    err = np.degrees(np.arccos(dot))
    return err

def compute_angular_error_stats(ang_err):
    """
    Angular error statistics such as min, max, mean, etc.

    Args:
        ang_err (np.array): angular error
    Returns:
        ang_err_stats (dict):  angular error statistics
    """
    ang_err = ang_err[~np.isnan(ang_err)]
    ang_err_stats = {"min": np.min(ang_err),
                     "10prc": np.percentile(ang_err, 10),
                     "median": np.median(ang_err),
                     "mean": np.mean(ang_err),
                     "90prc": np.percentile(ang_err, 90),
                     "max": np.max(ang_err)}
    return ang_err_stats



def main(shigehler_config):

    # Define the processed data path
    processedDataPath = "data/shigehler.npz"

    if os.path.isfile(processedDataPath):   
        # The file exists, load the data
        patches, patch_labels, image_idx = load_data(processedDataPath)
    else:
        dataPrepScript = "Data_preparation.py"
        try:
            # Start the subprocess
            process = subprocess.Popen(['python3', dataPrepScript])
            
            # Wait for the subprocess to finish
            process.wait()
        except subprocess.CalledProcessError as e:
            # Handle any errors or exceptions raised during the subprocess execution
            print(f"Subprocess failed with error code {e.returncode}")


    patches, patch_labels, image_idx = load_data("data/shigehler.npz")

    print(f"{patches.shape} {patch_labels.shape} {image_idx.shape}")
    model = IlluminantEstimationCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=0.0005)
    

    te_split = loadmat('assests/gehler_threefoldCVsplit.mat')['te_split'][0]

    npatches = patches.shape[0] # 56800
    npatchesPerImage = int(shigehler_config["SHIGEHLER"]["NPATCHES_PER_IMAGE"])
    nimages = npatches//npatchesPerImage #568


    TOTAL_RUN = 1

    cnn_patch_stats = []
    cnn_avg_pool_stats = []
    cnn_median_pool_stats = []

    cnn_patch_ang = np.zeros((npatches, TOTAL_RUN)) # 56800 x 1
    cnn_avg_pool_ang = np.zeros((nimages, TOTAL_RUN)) # 568 x 1
    cnn_median_pool_ang = np.zeros((nimages, TOTAL_RUN)) # 568 x 1

    # # due to randomness in neural networks, average over TOTAL_RUN runs
    for idx_run in range(TOTAL_RUN):
        
        cnn_est_patch_label = np.zeros((npatches, 3))
        cnn_est_avg_pool_label = np.zeros((nimages, 3))
        cnn_est_median_pool_label = np.zeros((nimages, 3))

        best_loss = np.inf
        for idx_fold in range(3): # 3-fold CV
            
            print("run {} fold {}".format(idx_run+1, idx_fold+1))
            te_idx = te_split[idx_fold][0] - 1 # since Matlab indexes start at 1, we subtract 1
            idx_test = np.in1d(image_idx, te_idx)
            idx_train = ~idx_test
            print(idx_test.sum())
            print(idx_train.sum())
            X_train, y_train = patches[idx_train,...], patch_labels[idx_train,:]
            X_test, y_test = patches[idx_test,...], patch_labels[idx_test,:]

            # Convert NumPy arrays to PyTorch tensors
            X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
            y_train = torch.from_numpy(y_train).float()
            X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
            y_test = torch.from_numpy(y_test).float()

            batch_size = 64
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            device = "mps" # set to cpu if fails
            model = IlluminantEstimationCNN().to(device)

            # Define your optimizer
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

            # Train your model
            num_epochs = 1
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)

                    # Calculate your custom Euclidean loss
                    loss = euclidean_distance_loss(targets, outputs)

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Calculate average loss for the epoch
                average_loss = total_loss / len(train_loader)

                print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss}")

            # Now, you can use the trained model to make predictions on your test data
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                y_pred = []
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)

                    loss = euclidean_distance_loss(targets, outputs)
                    
                    total_val_loss += loss.item()

                    y_pred.append(outputs)

                  
                avg_val_loss = total_val_loss / len(test_loader)
                print(f"Val Loss: {avg_val_loss}")

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
        
                y_pred = torch.cat(y_pred, dim=0)

            y_pred = y_pred.cpu().numpy()
            y_test = y_test.cpu().numpy()

            # generate a single illuminant estimation per image by pooling
            # the predicted patch illuminants
            for idx, idx_te in enumerate(te_idx):

                this_y_test = y_test[idx*npatchesPerImage:(idx+1)*npatchesPerImage]
                this_y_pred = y_pred[idx*npatchesPerImage:(idx+1)*npatchesPerImage]   
                cnn_est_patch_label[idx_te*npatchesPerImage:(idx_te+1)*npatchesPerImage, :] = this_y_pred
                cnn_est_avg_pool_label[idx_te, :] = np.mean(this_y_pred, axis=0)
                cnn_est_median_pool_label[idx_te, :] = np.median(this_y_pred, axis=0)
        
                cnn_patch_ang[idx_te*npatchesPerImage:(idx_te+1)*npatchesPerImage, idx_run] = compute_angular_error(this_y_test, this_y_pred)
            
            cnn_avg_pool_ang[te_idx, idx_run] = compute_angular_error(y_test[::npatchesPerImage,:], cnn_est_avg_pool_label[te_idx, :])
            cnn_median_pool_ang[te_idx, idx_run] = compute_angular_error(y_test[::npatchesPerImage,:], cnn_est_median_pool_label[te_idx, :])

        cnn_patch_stats.append(compute_angular_error_stats(cnn_patch_ang[:, idx_run]))
        cnn_avg_pool_stats.append(compute_angular_error_stats(cnn_avg_pool_ang[:, idx_run]))
        cnn_median_pool_stats.append(compute_angular_error_stats(cnn_median_pool_ang[:, idx_run]))

        print("Patch-wise angular error statistics")
        print(pd.DataFrame(cnn_patch_stats).mean())
        print("-----------------------------------")
        print()

        print("Average pooling angular error statistics")
        print(pd.DataFrame(cnn_avg_pool_stats).mean())
        print("-----------------------------------")
        print()

        print("Median pooling angular error statistics")
        print(pd.DataFrame(cnn_median_pool_stats).mean())

        np.savez("predictedAvgLabels.npz", cnn_est_avg_pool_label)
    

if __name__ == "__main__":
    shigehler_config = configparser.ConfigParser()
    shigehler_config.read("assests/shigehler.cfg")

    main(shigehler_config)
