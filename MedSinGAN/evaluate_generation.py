# Imports
import argparse
import logging
import os
import shutil

import torchvision.transforms as transforms
from PIL import Image
from cleanfid import fid
from piqa import SSIM, MS_SSIM, LPIPS


def lpips_test(original, generated):
    """
    Performs the lpips test on the generated images
    @param original: Original image
    @param generated: Generated images
    @return: Logs the information in the log file
    """

    logging.info('===== LPIPS TESTS =====')

    # Initialize criterion
    criterion = LPIPS()

    # Calculate average
    average = 0

    # For each generated image calculate lpips
    for image in generated.keys():
        loss = criterion(original, generated[image])
        average += loss
        logging.info(f'Image: {image}, LPIPS (Alex): {loss}')

    logging.info(f'Average LPIPS: {average / len(generated.keys())}')


def ms_ssim_test(original, generated):
    """
    Performs MS-SSIM metric in the generated images
    @param original: Original image
    @param generated: Generated set of images
    @return: Logs the scores on the log file
    """

    logging.info('===== MS-SSIM TESTS =====')

    ssim = SSIM()
    msssim = MS_SSIM()

    # Calculate average
    average_ssim = 0
    average_mssim = 0

    # For each generated image calculate MS-SSIM and SSIM
    for image in generated.keys():
        ssim_loss = ssim(original, generated[image])
        average_ssim += ssim_loss

        msssim_loss = msssim(original, generated[image])
        average_mssim += msssim_loss
        logging.info(f'Image: {image}, SSIM: {ssim_loss}, MS-SSIM: {msssim_loss}')

    logging.info(
        f'Average SSIM: {average_ssim / len(generated.keys())}, '
        f'Average MS-SSIM: {average_mssim / len(generated.keys())}')


def sifid_test(original_image, generated_folder):
    """
    Calculates the SIFID score between the origin folder and the generated one
    @param original_image: Original image
    @param generated_folder: Folder with N generated images
    @return: Logs the score in the log file
    """

    logging.info('===== SIFID TESTS =====')

    # Generate folder with N copies of the original image
    folder_name = 'original_images_folder'
    os.mkdir(folder_name)
    imag = Image.open(original_image)

    # Generate folder for original images
    for i in range(len(os.listdir(generated_folder))):
        imag.save(folder_name + '/original_' + str(i) + '.jpg')

    # Compute fid's
    score = fid.compute_fid(folder_name, generated_folder)
    logging.info(f'Image: {im}, SIFID: {score}')

    # Delete temporary folder
    shutil.rmtree(folder_name)


if __name__ == '__main__':

    logging.basicConfig(filename='generation_evaluation.log', filemode='a', level=logging.INFO, format='%(message)s')

    # Receive arguments to then evaluate metrics
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', help='Real image. Ex: images/photo.png', required=True)
    parser.add_argument('--output_folder', help='Complete path folder with images to test', required=True)

    opt = parser.parse_args()

    """
    1. Read output images
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    logging.info(f'Reading output images from {opt.output_folder}')

    output_images = {}
    for im in os.listdir(opt.output_folder):
        output_images[im] = transform(Image.open(opt.output_folder + '/' + im))
        output_images[im] = output_images[im].reshape(
            (1, output_images[im].shape[0], output_images[im].shape[1], output_images[im].shape[2]))

    """
    2. Convert Input Image to RGB if not
    """
    logging.info(f'Reading original image {opt.input_image}')
    gray_rgb = transform(Image.open(opt.input_image))
    gray_rgb = gray_rgb.reshape((1, gray_rgb.shape[0], gray_rgb.shape[1], gray_rgb.shape[2]))

    """
    3. Run LPIPS tests
    """
    lpips_test(gray_rgb, output_images)

    """
    4. Run MS-SSIM tests
    """
    ms_ssim_test(gray_rgb, output_images)

    """
    5. Run SIFID tests
    """
    sifid_test(opt.input_image, opt.output_folder)

    """
    6. Adds space for posteriours LOGS
    """
    logging.info('\n\n\n')
