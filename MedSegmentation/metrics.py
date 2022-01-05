import numpy as np
from keras import backend as K


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0


def dice_coeff(y_true, y_pred, smooth=1):
    """
    Calculates the dice coefficient for the images
    @param y_true: The ground truth mask
    @param y_pred: The predicted mask
    @param smooth:
    @return: The Dice Coefficient value
    """

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return K.get_value(dice)


def dice_coeff_variant(y_true, y_pred, smooth=1):
    """
    Calculates the dice coefficient for the images
    @param y_true: The ground truth mask
    @param y_pred: The predicted mask
    @param smooth:
    @return: The Dice Coefficient value
    """

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = K.mean((2. * intersection + smooth) / (union + smooth))
    return K.get_value(dice)


def jaccard_index(y_true, y_pred, smooth=1):
    """
    Performs jaccard index using jaccard score from scikit learn
    @param smooth:
    @param y_true: The true ground truth value
    @param y_pred: The predicted value
    @return: The Jaccard score between the two images
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return K.get_value(iou)


def jaccard_index_variant(y_true, y_pred, smooth=1):
    """
    Performs jaccard index using jaccard score from scikit learn
    @param smooth:
    @param y_true: The true ground truth value
    @param y_pred: The predicted value
    @return: The Jaccard score between the two images
    """

    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return K.get_value(iou)
