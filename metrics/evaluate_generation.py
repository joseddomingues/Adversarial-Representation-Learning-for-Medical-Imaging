# Imports
import argparse
import logging
import os

import cv2
import lpips
import torch
from cleanfid import fid
from piqa import SSIM, MS_SSIM

import utils

logging.basicConfig(filename='evaluation.log', filemode='a', level=logging.INFO, format='%(message)s')
# Receive arguments to then evaluate metrics

parser = argparse.ArgumentParser()

parser.add_argument('--input_image', help='Real image. Ex: images/photo.png', required=True)
parser.add_argument('--output_folder', help='Complete path folder with images to test', required=True)

opt = parser.parse_args()
"""
1. Read output images
"""
logging.info(f'Reading output images from {opt.output_folder}')

output_images = {}
for im in os.listdir(opt.output_folder):
    output_images[im] = cv2.imread(opt.output_folder + '/' + im, 0)
    output_images[im] = utils.convert_to_rgb(output_images[im])

"""
2. Convert Input Image to RGB if not
"""

logging.info(f'Reading original image {opt.input_image}')

gray_rgb = cv2.imread(opt.input_image, 0)
gray_rgb = utils.convert_to_rgb(gray_rgb)

"""
3. Runs LPIPS tests
"""

logging.info('===== LPIPS TESTS =====')

# Losses
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

# Normalize images
gray_rgb_normalized = utils.normalize_rgb_image(gray_rgb)

for im in output_images.keys():
    break
    current = utils.normalize_rgb_image(output_images[im])
    alex_loss = loss_fn_alex(current, gray_rgb_normalized)
    vgg_loss = loss_fn_vgg(current, gray_rgb_normalized)
    logging.info(f'Image: {im}, LPIPS(Alex): {alex_loss}, LPIPS(Vgg): {vgg_loss}')

"""
4. Run MS-SSIM tests
"""

logging.info('\n\n===== MS-SSIM TESTS =====')

ssim = SSIM()
msssim = MS_SSIM()
gray_rgb_tensor = torch.tensor(utils.normalize_rgb_image_standard(gray_rgb)).reshape(1, 3, gray_rgb.shape[0],
                                                                                     gray_rgb.shape[1])

for im in output_images.keys():
    break
    current = torch.tensor(utils.normalize_rgb_image_standard(output_images[im])).reshape(1, 3,
                                                                                          output_images[im].shape[0],
                                                                                          output_images[im].shape[1])
    ssim_loss = ssim(current, gray_rgb_tensor)
    msssim_loss = msssim(current, gray_rgb_tensor)
    logging.info(f'Image: {im}, SSIM Loss: {ssim_loss}, MS-SSIM Loss: {msssim_loss}')

"""
5. Run SIFID tests
"""

logging.info('\n\n===== SIFID TESTS =====')
torch.device
score = fid.compute_fid(opt.output_folder, opt.output_folder)
logging.info(f'Image: {im}, SSIM Loss: {ssim_loss}, MS-SSIM Loss: {msssim_loss}')

"""
6. Run NIQE tests
"""
for im in output_images.keys():
    break
    current = utils.normalize_rgb_image(output_images[im])
    logging.info(f'Image: {im}, NIQE: {niqe.niqe(curent)}')

# img = scipy.misc.imread(sys.argv[1], flatten=True).astype(numpy.float)/255.0
