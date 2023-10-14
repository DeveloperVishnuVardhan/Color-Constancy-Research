"""
CS-7180 Fall 2023
Authors:
1. Jyothi Vishnu Vardhan Kolla
2. Karan Shah

This file contain the code to the utility functions used in the project.
"""

import glob
import numpy as np
from scipy.io import loadmat
import numpy as np


def get_images_full_path(img_folder_path):
    """ get all images for the specified folder.

    Args:
        img_folder_path (_type_): Path which contains the Images.
    Returns:
        images_full_path: A list that contains path of all Images.
    """

    images_full_path = glob.glob(
        img_folder_path + "/**/*.png", recursive=True)
    images_full_path.sort(key=lambda x: x.split('/')[-1].split('.')[0])
    return images_full_path


def load_groundtruth_illuminant(file_path):
    """ load ground truth illuminant

    Args:
        file_path (str): path which contains the ground truth illuminant values
    Returns:
        real_rgb (np.array):  ground truth illuminant values
    """

    real_illum_568 = loadmat(file_path)
    real_rgb = real_illum_568["real_rgb"]
    real_rgb = real_rgb / real_rgb[:, 1][:,
                                         np.newaxis]  # convert to chromaticity
    return real_rgb

def load_data(data_path):
    """Loads and returns the preprocessed data.

    Args:
        data_path (str): path of the dataset.
    Returns:
        returns the patches, patch_labels, image_idx
    """
    dataset = np.load(data_path)
    patches = dataset['data']
    patch_labels = dataset['patch_labels']
    image_idx = dataset['image_idx']

    return patches, patch_labels, image_idx


