import os
import shutil

import cv2
import imageio


def extract_harmonized_samples(input_folder="/Users/josedaviddomingues/Desktop/harmonised_samples/16_stages_malign",
                               new_folder_name="/Users/josedaviddomingues/Desktop/malign_harmonised"):
    """
    Extract harmonised samples from input folder
    @param input_folder: The input folder path
    @param new_folder_name: The new folder name
    @return: -
    """

    os.mkdir(new_folder_name)

    folds_root = os.listdir(input_folder)
    try:
        folds_root.remove('.DS_Store')
    except:
        print('No DS in the list')

    for base_folder in folds_root:
        folds = os.listdir(input_folder + "/" + base_folder)
        try:
            folds.remove('.DS_Store')
        except:
            print('No DS in the list')
        for image in folds:
            if 'harmonized' in image:
                shutil.move(input_folder + "/" + base_folder + '/' + image,
                            new_folder_name + '/' + base_folder + '_' + image)


def generate_gif(images_folder='/images', gif_name='out_images'):
    """
    Generate a gif from input folder images
    @param images_folder: Images folder path
    @param gif_name: Gif name (without .gif)
    @return: -
    """
    
    images_gif = []
    for imag in os.listdir(images_folder):
        images_gif.append(imageio.imread(images_folder + '/' + imag))
    imageio.mimsave(images_folder + '/' + gif_name + '.gif', images_gif)


def convert_gray_to_rgb(image_path='image.png', output_name='image_rgb.png'):
    """
    Converts image to rgb gray-scale
    @param image_path: Image to convert
    @param output_name: Image output name
    @return: -
    """

    gray = cv2.imread(image_path, 0)
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_name, backtorgb)
