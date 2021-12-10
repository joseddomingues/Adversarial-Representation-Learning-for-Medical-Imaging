# Imports
import os
import shutil

import torchvision.transforms as transforms
from PIL import Image
from cleanfid import fid
from piqa import SSIM, MS_SSIM, LPIPS


class GenerationEvaluator:
    def __init__(self, input_image, generated):
        self.original_image = input_image
        self.generated_images = generated

        # Required for FID
        self.base_image = input_image
        self.output_folder = generated

        # 1. Read output images
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        output_images = {}
        for im in os.listdir(self.generated_images):
            output_images[im] = transform(Image.open(self.generated_images + '/' + im))
            output_images[im] = output_images[im].reshape(
                (1, output_images[im].shape[0], output_images[im].shape[1], output_images[im].shape[2]))

        self.generated_images = output_images

        # 2. Convert Input Image to RGB if not
        self.original_image = transform(Image.open(self.original_image))
        self.original_image = self.original_image.reshape(
            (1, self.original_image.shape[0], self.original_image.shape[1], self.original_image.shape[2]))

    def run_lpips(self):
        """
        Run LPIPS test for the generation
        @return: The average LPIPS value for all generated images
        """

        # Initialize criterion
        criterion = LPIPS()

        # Calculate average
        average = 0

        # For each generated image calculate lpips
        for image in self.generated_images.keys():
            loss = criterion(self.original_image, self.generated_images[image])
            average += loss

        return average / len(self.generated_images.keys())

    def run_mssim(self):
        """
        Run SSIM and MS-SSIM test for the generation
        @return: SSIM, MS-SSIM
        """

        ssim = SSIM()
        msssim = MS_SSIM()

        # Calculate average
        average_ssim = 0
        average_mssim = 0

        # For each generated image calculate MS-SSIM and SSIM
        for image in self.generated_images.keys():
            ssim_loss = ssim(self.original_image, self.generated_images[image])
            average_ssim += ssim_loss

            msssim_loss = msssim(self.original_image, self.generated_images[image])
            average_mssim += msssim_loss

        return average_ssim / len(self.generated_images.keys()), average_mssim / len(self.generated_images.keys())

    def run_fid(self):
        """
        Run the FID test for the generated images
        @return: FID value
        """

        # Generate folder with N copies of the original image
        folder_name = 'original_images_folder'
        os.mkdir(folder_name)
        imag = Image.open(self.base_image)

        # Generate folder for original images
        for i in range(len(os.listdir(self.output_folder))):
            imag.save(folder_name + '/original_' + str(i) + '.jpg')

        # Compute fid's
        score = fid.compute_fid(folder_name, self.output_folder)

        # Delete temporary folder
        shutil.rmtree(folder_name)

        return score
