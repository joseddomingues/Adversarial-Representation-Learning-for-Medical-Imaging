from keras import backend as K


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
