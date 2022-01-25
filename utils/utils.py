# Import required libraries
import cv2
import os
import shutil
import imageio
from skimage import io as img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision as tv
from PIL import Image


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


def get_latest_model(base_path):
    """
    Get the latest model path
    @param base_path: base model path
    @return: The latest model path
    """
    models = os.listdir(base_path)

    latest = 0  # Values will always be bigger than 0
    desired = models[0]

    for id, model in enumerate(models):
        splitted = model.split("_")
        code = splitted[:6]
        code = int(''.join(code))
        if code > latest:
            latest = code
            desired = model

    return os.path.join(base_path, desired)


###################################################################################################

def get_image_laterality(image):
    left_edge = np.sum(image[:, 0])
    right_edge = np.sum(image[:, -1])
    return (True, False) if left_edge < right_edge else (False, True)


def get_measures(image):
    positions = np.nonzero(image)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    return top, right, bottom, left


def get_start_coordinate(image):
    positions = np.nonzero(image)
    bottom = positions[0].max()
    x_bottom = int(np.mean(np.nonzero(image[bottom])))
    return x_bottom, bottom


def get_correct_value(number):
    if number == 0:
        return 0
    else:
        return 1


def image_to_binary(image, pth):
    b_image = []
    for arr in image:
        curr = [get_correct_value(elem) for elem in arr]
        b_image.append(curr)
    b_image = np.array(b_image, dtype=np.uint8)

    plt.imsave(pth, np.array(b_image), cmap=cm.gray)
    return b_image


def does_collage_mask(width, height, malign, normal):
    # Crop both the mass, and the normal
    crop_segmentation(malign, 'malign_aux.png')

    normal_image = Image.open(normal)
    mass_to_paste = Image.open('malign_aux.png')

    # Creates collage and save
    back_im = normal_image.copy()
    back_im.paste(mass_to_paste, (width, height), mass_to_paste)

    return list(back_im.getdata()) == list(normal_image.getdata())


def is_collage_possible(malign_mask_pth, normal_breast_pth):
    # Operations Threshold
    threshold = 50

    # Read the images
    malign_mask = cv2.imread(malign_mask_pth, cv2.IMREAD_GRAYSCALE)
    normal_breast = cv2.imread(normal_breast_pth, cv2.IMREAD_GRAYSCALE)
    _, normal_x = normal_breast.shape
    normal_breast = image_to_binary(normal_breast, '/content/normal_aux.png')

    # Get images laterality
    R, _ = get_image_laterality(normal_breast)

    # Get images measures
    # Calculate malign mass measures
    m_top, m_right, m_bottom, m_left = get_measures(malign_mask)

    # Calculate normal breast measures
    n_top, n_right, n_bottom, n_left = get_measures(normal_breast)

    # Calculate widths and heights
    malign_mass_width = abs(m_right - m_left)
    malign_mass_height = abs(m_bottom - m_top)
    normal_breast_width = abs(n_right - n_left)
    normal_breast_height = abs(n_bottom - n_top)

    # Check if its worth the try
    if malign_mass_width > normal_breast_width or malign_mass_height > normal_breast_height:
        return -1, -1

    # Get bottom base coordinate
    bottom_coordinate = get_start_coordinate(normal_breast)

    # Coordinate collage starts bottom
    c, d = bottom_coordinate

    if R:

        # Check if mass is all inside image. If not, then go left + threshold
        if normal_x - c < malign_mass_width:
            c, d = c - (malign_mass_width - (normal_x - c) + threshold), d

        # Go up the height plus the threshold
        c, d = c, d - (malign_mass_height + threshold)

        # Go up until the masks match. If never match then skip them
        while d > threshold:
            if does_collage_mask(c, d, malign_mask_pth, '/content/normal_aux.png'):
                return c, d

            c, d = c, d - threshold

        return -1, -1
    else:

        # Check if mass is all inside image. If not, then go right + threshold
        if c < malign_mass_width:
            c, d = c + (malign_mass_width - c + threshold), d

        # Go up the height plus the threshold
        c, d = c, d - (malign_mass_height + threshold)

        # Go up until the masks match. If never match then skip them
        while d > threshold:
            if does_collage_mask(c, d, malign_mask_pth, '/content/normal_aux.png'):
                return c, d

            c, d = c, d - threshold

        return -1, -1


# Remove the 4 channel to collage image
def remove_4_channel(im_path, output_path):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    # Transpose naive image to properly see it
    tranposed = img.transpose(2, 0, 1)

    # Transpose image again with only the 3 rgb channels to save
    output = tranposed[0:3].transpose(1, 2, 0)

    # Save new naive image (3-channels)
    cv2.imwrite(output_path, output)


# Resize image for hamronisation
def resize_image(im_path, percent_original, output_path):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ', img.shape)

    scale_percent = percent_original  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)
    cv2.imwrite(output_path, resized)


# Make mask have 3 channels
def make_3_channels_mask(im_path, out_path):
    i = img.imread(im_path)
    new_i = []
    new_i.append(i)
    new_i.append(i)
    new_i.append(i)
    new_i = torch.tensor(np.array(new_i))
    tv.io.write_png(new_i, out_path)


# Crops the segmentation by its limits
def crop_segmentation(fp, outp):
    imag = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    imageObject = Image.open(fp)
    positions = np.nonzero(imag)

    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()

    cropped = imageObject.crop((left, top, right, bottom))
    cropped.save(outp)


# Makes a collage given the malign image, the malign mask, and the normal image
def make_collage(malign_pth, malign_mask_pth, normal_pth, width, height):
    # Reads malign base image
    malign = cv2.imread(malign_pth, cv2.IMREAD_UNCHANGED)

    # Convert mask to 3 channels
    make_3_channels_mask(malign_mask_pth, '/content/malign_mask3.png')
    malign_mask = cv2.imread('/content/malign_mask3.png', cv2.IMREAD_UNCHANGED)

    # Grab the image mask from the mass image
    masked = malign.copy()
    masked[malign_mask == 0] = 0
    cv2.imwrite('/content/segmented_mass.png', masked)

    # Crop both the mask, and the masked mass
    crop_segmentation('/content/segmented_mass.png', '/content/cropped_mass.png')
    crop_segmentation(malign_mask_pth, '/content/malign_mask_cropped.png')

    normal_image = Image.open(normal_pth)
    mass_to_paste = Image.open('/content/cropped_mass.png')
    mass_mask = Image.open('/content/malign_mask_cropped.png')

    # Creates collage and save
    back_im = normal_image.copy()
    back_im.paste(mass_to_paste, (width, height), mass_mask)
    back_im.save('/content/collage.png', quality=95)

    # Creates collage mask
    collage_mask = Image.new("L", back_im.size, 0)
    collage_mask.paste(mass_mask, (width, height))
    collage_mask.save('/content/collage_mask.png', quality=95)

    # Deletes unecessary images
    try:
        os.remove('/content/malign_mask3.png')
        os.remove('/content/segmented_mass.png')
        os.remove('/content/cropped_mass.png')
        os.remove('/content/malign_mask_cropped.png')
    except OSError as e:
        print(f"FAILED\nFile: {e.filename}\nError: {e.strerror}")
