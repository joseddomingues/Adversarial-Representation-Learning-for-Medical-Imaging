import argparse
import logging
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
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


if __name__ == '__main__':

    logging.basicConfig(filename='segmentation_evaluation.log', filemode='a', level=logging.INFO, format='%(message)s')

    # Receive arguments to then evaluate metrics
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', help='Real image. Ex: images', required=True)
    parser.add_argument('--output_folder', help='Complete path folder with images to test', required=True)

    opt = parser.parse_args()

    """
    1. Read input/output images
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Input Images
    logging.info(f'Reading input images from {opt.input_folder}')

    input_images = {}
    for im in os.listdir(opt.input_folder):
        input_images[im] = transform(Image.open(opt.input_images + '/' + im))
        input_images[im] = input_images[im].reshape(
            (1, input_images[im].shape[0], input_images[im].shape[1], input_images[im].shape[2]))

    # Output Images
    logging.info(f'Reading output images from {opt.output_folder}')

    output_images = {}
    for im in os.listdir(opt.output_folder):
        output_images[im] = transform(Image.open(opt.output_folder + '/' + im))
        output_images[im] = output_images[im].reshape(
            (1, output_images[im].shape[0], output_images[im].shape[1], output_images[im].shape[2]))

    """
    2. Convert Input Image to RGB if not
    """
    # logging.info(f'Reading original image {opt.input_image}')
    # gray_rgb = transform(Image.open(opt.input_image))
    # gray_rgb = gray_rgb.reshape((1, gray_rgb.shape[0], gray_rgb.shape[1], gray_rgb.shape[2]))

    """
    3. Run Pixel Accuracy
    """
    avg_pixel_acc = 0
    for im in input_images.keys():
        avg_pixel_acc += pixel_accuracy(input_images[im], output_images[im.replace('.png', '_mask.png')])
    logging.info(f'Average Pixel Accuracy Segmentation: {avg_pixel_acc / len(input_images.keys())}')

    """
    4. Run Jaccard Index
    """
    avg_jaccard_index = 0
    for im in input_images.keys():
        avg_jaccard_index += jaccard_index(input_images[im], output_images[im.replace('.png', '_mask.png')])
    logging.info(f'Average Jaccard Index Segmentation: {avg_jaccard_index / len(input_images.keys())}')

    """
    5. Run Dice coefficient 
    """
    avg_dice_coef = 0
    for im in input_images.keys():
        avg_dice_coef += dice_coeff(output_images[im.replace('.png', '_mask.png')], input_images[im])
    logging.info(f'Average Jaccard Index Segmentation: {avg_dice_coef / len(input_images.keys())}')

    """
    6. Adds space for posteriours LOGS
    """
    logging.info('\n\n\n')
