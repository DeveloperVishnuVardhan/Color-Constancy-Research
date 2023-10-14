"""
CS-7180 Fall 2023
Authors:
1. Jyothi Vishnu Vardhan Kolla
2. Karan Shah

Date: 09/14/2023
This file contain the code to Preprocess the images.
"""

import cv2
import numpy as np


def resize_image(image, maxwh=1200):
    """Resizes images such that max(w, h) = 1200.

    Args:
        image (_type_): Input Image to be resized.
        maxwh (int, optional): _description_. Defaults to 1200.
    Returns:
        image_resized (np.array): output resized image.
    """

    r, c, _ = image.shape
    image_scale = maxwh / np.maximum(r, c)
    image_resized = cv2.resize(
        image, (int(np.ceil(c*image_scale)), int(np.ceil(r*image_scale))))
    return image_resized


def hist_stretch(img, bit_depth):
    """
    Contrast normalization via global histogram stretching

    Args:
        img (np.array): Input image
        bit_depth (int): Number of bits per channel
    Returns:
        img_out (np.array): Contast normalized image
    """

    if np.max(img) - np.min(img) < 1e-5:  # Do-nothing
        img_out = img
    else:
        min_img, max_img = np.min(img), np.max(img)
        img_out = ((img - min_img) / (max_img - min_img)) * \
            (2 ** bit_depth - 1)

    return img_out
