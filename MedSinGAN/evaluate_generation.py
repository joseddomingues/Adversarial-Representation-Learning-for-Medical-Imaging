# Imports
import os
import shutil

import torchvision.transforms as transforms
from PIL import Image
from cleanfid import fid
from piqa import SSIM, MS_SSIM, LPIPS

MIN_SSIM_SIZE = 256


class GenerationEvaluator:
    def __init__(self, input_image, generated=None, padd_input=False, adjust_sizes=False):
        self.original_image_path = input_image
        self.original_image = input_image
        self.generated_images = generated

        # Required for FID
        self.base_image = input_image
        self.output_folder = generated

        # Create transformation function to convert PIL images to tensors
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 1. Read output images
        if generated:
            output_images = {}
            for im in os.listdir(self.generated_images):
                output_images[im] = transform(Image.open(os.path.join(self.generated_images, im)))
                output_images[im] = output_images[im].reshape(
                    (1, output_images[im].shape[0], output_images[im].shape[1], output_images[im].shape[2]))

            self.generated_images = output_images

        # 2. Convert Input Image to RGB if not
        self.original_image = Image.open(self.original_image)
        if padd_input:
            new_w = int(MIN_SSIM_SIZE / min(self.original_image.size) * self.original_image.size[0])
            new_h = int(MIN_SSIM_SIZE / min(self.original_image.size) * self.original_image.size[1])
            self.original_image = self.original_image.resize((new_w, new_h))

        if adjust_sizes and self.generated_images:
            example_sample = list(self.generated_images.keys())[0]
            example_width = self.generated_images[example_sample].shape[2]
            example_height = self.generated_images[example_sample].shape[3]
            self.original_image = self.original_image.resize((example_height, example_width), Image.ANTIALIAS)

        self.original_image = transform(self.original_image)
        self.original_image = self.original_image.reshape(
            (1, self.original_image.shape[0], self.original_image.shape[1], self.original_image.shape[2]))

    def run_lpips(self):
        """
        Run LPIPS test for the generation
        @return: The average LPIPS value for all generated images
        """

        # Initialize criterion
        criterion = LPIPS()

        # If not valid folder
        if not self.generated_images:
            return -1

        # Calculate average
        average = 0

        # For each generated image calculate lpips
        for image in self.generated_images.keys():
            loss = criterion(self.original_image, self.generated_images[image])
            average += loss

        return (average / len(self.generated_images.keys())).item()

    def run_lpips_to_image(self, generated_image, padd=False):
        """
        Run LPIPS test for the given generated image
        @param generated_image: Generated image to lpips
        @return: The average LPIPS value for the generated image
        """

        # If image doesnt exists
        if not os.path.exists(generated_image):
            return -1

        # Initialize criterion
        criterion = LPIPS()

        # Transformation for generated
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        gen = Image.open(generated_image)
        if padd:
            new_w = int(MIN_SSIM_SIZE / min(gen.size) * gen.size[0])
            new_h = int(MIN_SSIM_SIZE / min(gen.size) * gen.size[1])
            gen = gen.resize((new_w, new_h))
        gen = transform(gen)
        gen = gen.reshape(
            (1, gen.shape[0], gen.shape[1], gen.shape[2]))

        curr_ori = Image.open(self.original_image_path)
        curr_ori = curr_ori.resize((gen.shape[3], gen.shape[2]), Image.ANTIALIAS)
        curr_ori = transform(curr_ori)
        curr_ori = curr_ori.reshape(
            (1, curr_ori.shape[0], curr_ori.shape[1], curr_ori.shape[2]))

        return criterion(curr_ori, gen)

    def run_mssim(self):
        """
        Run SSIM and MS-SSIM test for the generation
        @return: SSIM, MS-SSIM
        """

        ssim = SSIM()
        msssim = MS_SSIM()

        # If not valid folder
        if not self.generated_images:
            return -1

        # Calculate average
        average_ssim = 0
        average_mssim = 0

        # For each generated image calculate MS-SSIM and SSIM
        for image in self.generated_images.keys():
            ssim_loss = ssim(self.original_image, self.generated_images[image])
            average_ssim += ssim_loss

            msssim_loss = msssim(self.original_image, self.generated_images[image])
            average_mssim += msssim_loss

        return (average_ssim / len(self.generated_images.keys())).item(), \
               (average_mssim / len(self.generated_images.keys())).item()

    def run_mssim_to_image(self, generated_image, padd=False):
        """
        Run SSIM and MS-SSIM test for the given generated image
        @param generated_image: Generated image to mssim
        @return: SSIM, MS-SSIM
        """

        # If image doesnt exists
        if not os.path.exists(generated_image):
            return -1

        ssim = SSIM()
        msssim = MS_SSIM()

        # Transformation for generated
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        gen = Image.open(generated_image)
        if padd:
            new_w = int(MIN_SSIM_SIZE / min(gen.size) * gen.size[0])
            new_h = int(MIN_SSIM_SIZE / min(gen.size) * gen.size[1])
            gen = gen.resize((new_w, new_h))
        gen = transform(gen)
        gen = gen.reshape(
            (1, gen.shape[0], gen.shape[1], gen.shape[2]))

        curr_ori = Image.open(self.original_image_path)
        curr_ori = curr_ori.resize((gen.shape[3], gen.shape[2]), Image.ANTIALIAS)
        curr_ori = transform(curr_ori)
        curr_ori = curr_ori.reshape(
            (1, curr_ori.shape[0], curr_ori.shape[1], curr_ori.shape[2]))

        return ssim(curr_ori, gen), msssim(curr_ori, gen)

    def run_fid(self):
        """
        Run the FID test for the generated images
        @return: FID value
        """

        # If not valid folder
        if not self.generated_images:
            return -1

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
