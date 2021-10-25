import cv2
import numpy as np


def convert_to_rgb(image):
    """
    Converts image to RGB if its not already
    @param image: Input image
    @return: Returns the image RGB
    """
    if len(image.shape) != 3 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def normalize_rgb_image(image):
    """
    Normalize rgb image to scale [-1,1]
    @param image: Image to normalize
    @return: The normalized image
    """
    normalized_input = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    return 2 * normalized_input - 1


def normalize_rgb_image_standard(image):
    """
    Normalize rgb image to scale [0,1]
    @param image: Image to normalize
    @return: The normalized image
    """
    return (image - np.amin(image)) / (np.amax(image) - np.amin(image))
