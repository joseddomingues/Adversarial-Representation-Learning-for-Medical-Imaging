import numpy as np
from sklearn.metrics import jaccard_score


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    # print(im_sum)

    return 2. * intersection.sum() / im_sum


def jaccard_index(true, pred):
    """
    Performs jaccard index using jaccard score from scikit learn
    @param true: The true ground truth value
    @param pred: The predicted value
    @return: The Jaccard score between the two images
    """

    return jaccard_score(true.flatten(), pred.flatten())


def pixel_accuracy(eval_segm, gt_segm):
    """
    Performs pixel accuracy metric for segmentation
    @param eval_segm: Predicted segmentation
    @param gt_segm: Ground truth segmentation
    @return: The Pixel accuracy segmentation
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if sum_t_i == 0:
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    """

    @param eval_segm:
    @param gt_segm:
    @param cl:
    @param n_cl:
    @return:
    """
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    """

    @param segm:
    @return:
    """
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    """

    @param segm:
    @param cl:
    @param n_cl:
    @return:
    """
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    """

    @param segm:
    @return:
    """
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    """

    @param eval_segm:
    @param gt_segm:
    @return:
    """
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


class EvalSegErr(Exception):
    """

    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


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
