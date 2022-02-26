# Import required libraries
import os
import shutil
import subprocess

import cv2
import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
from PIL import Image, ImageChops
from skimage import io as img


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


def crop_folder(folder_path, itype, target_path):
    """
    Crops a folder images to improve training quality
    @param folder_path: Folder to crop
    @param itype: Type of images. One of ("N", "M", "B", "A")
    @param target_path: Target folder
    @return:
    """
    if itype == "N" or itype == "A":
        for image_elem in os.listdir(folder_path):

            if ".DS_Store" in image_elem:
                continue

            crop_segmentation(os.path.join(folder_path, image_elem), os.path.join(target_path, image_elem))

    else:
        for elem in os.listdir(folder_path):

            if ".DS_Store" in elem or "_mask" in elem:
                continue

            imag = cv2.imread(os.path.join(folder_path, elem), cv2.IMREAD_UNCHANGED)
            imageObject = Image.open(os.path.join(folder_path, elem))
            imageObject_mask = Image.open(os.path.join(folder_path, elem.replace(".png", "_mask.png")))
            positions = np.nonzero(imag)

            top = positions[0].min()
            bottom = positions[0].max()
            left = positions[1].min()
            right = positions[1].max()

            cropped = imageObject.crop((left, top, right, bottom))
            cropped_mask = imageObject_mask.crop((left, top, right, bottom))
            cropped.save(os.path.join(target_path, elem))
            cropped_mask.save(os.path.join(target_path, elem.replace(".png", "_mask.png")))


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

    for _, model in enumerate(models):
        splitted = model.split("_")
        code = splitted[:6]
        code = int(''.join(code))
        if code > latest:
            latest = code
            desired = model

    return os.path.join(base_path, desired)


def execute_bash_command(cmd):
    """
    Executes a specified command and outputs the result of the training progress
    @param cmd: Command to execute
    @return: The lines of the execution command
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def get_image_core_name(image_name):
    """
    Obtains the image core name. Everything before the .png|.jpg etc and after /
    @param image_name: Image name
    @return: The image core name
    """
    core_name = image_name.split("/")[-1]
    core_name = core_name.split(".")[:-1]
    core_name = ".".join(core_name)
    return core_name


############################## HARMONISATION PROCESSING TECHNIQUES ###########################################


def crop_images_to_same_size(image_arr):
    """
    Crop the images in the dimensions of the biggest crop
    @param image_arr: Image array with all the images paths
    @return: -
    """
    # Base values to crop
    # No image has 1.000.000 million pixels so far so its good
    top = 1000000
    bottom = -1
    left = 1000000
    right = -1

    for elem in image_arr:
        curr = cv2.imread(elem, cv2.IMREAD_UNCHANGED)
        positions = np.nonzero(curr)
        curr_top = positions[0].min()
        curr_bottom = positions[0].max()
        curr_left = positions[1].min()
        curr_right = positions[1].max()

        if curr_top < top:
            top = curr_top
        if curr_bottom > bottom:
            bottom = curr_bottom
        if curr_left < left:
            left = curr_left
        if curr_right > right:
            right = curr_right

    for elem in image_arr:
        image_object = Image.open(elem)
        cropped = image_object.crop((left, top, right, bottom))
        os.remove(elem)
        cropped.save(elem)


def get_image_laterality(image):
    """
    Get image laterality
    @param image: Image to find laterality
    @return: "(R,L)" -> For veracity values
    """
    left_edge = np.sum(image[:, 0])
    right_edge = np.sum(image[:, -1])
    return (True, False) if left_edge < right_edge else (False, True)


def get_measures(image):
    """
    Get image widest positions
    @param image: Image to return positions
    @return: top, right, bottom, left point
    """
    positions = np.nonzero(image)
    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()
    return top, right, bottom, left


def get_start_coordinate(image, mass_path, laterality_r):
    """
    Get image start coordinate collage point
    @param image: Image to get start coordinate
    @param mass_path: Mass path
    @param laterality_r: Laterality of the image (True if R | False if L)
    @return: The starting coordinate
    """
    imag = cv2.imread(mass_path, cv2.IMREAD_UNCHANGED)
    positions = np.nonzero(image)
    left = positions[1].min()
    right = positions[1].max()
    vertical_co = positions[0][list(positions[1]).index(left)]
    vertical_co_r = positions[0][list(positions[1]).index(right)]

    if laterality_r:
        return left, int(vertical_co - imag.shape[1] / 2)
    else:
        return right, int(vertical_co_r - imag.shape[1] / 2)


def get_correct_value(number):
    """
    Auxiliary function to convert image to binary values
    @param number: Number to check
    @return: 0 | 255 according to the value
    """
    if number == 0:
        return 0
    else:
        return 255


def image_to_binary(image, pth):
    """
    Convert image to binary
    @param image: Image to convert image
    @param pth: Path to save the image
    @return: The image in binary
    """
    b_image = []
    for arr in image:
        curr = [get_correct_value(elem) for elem in arr]
        b_image.append(curr)
    b_image = np.array(b_image, dtype=np.uint8)

    plt.imsave(pth, np.array(b_image), cmap=cm.gray)
    return b_image


def does_collage_mask(width, height, normal):
    """
    Verifies if this try of collage is possible
    @param width: Width of starting point
    @param height: Height of starting point
    @param normal: Normal breast
    @return: True if possible | False if not
    """
    normal_image = Image.open(normal)
    normal_image = Image.fromarray(np.array(normal_image)[:, :, 0])
    mass_to_paste = Image.open('malign_aux.png')

    # Creates collage and save
    back_im = normal_image.copy()
    back_im.paste(mass_to_paste, (width, height), mass_to_paste)

    if ImageChops.difference(back_im, normal_image).getbbox():
        return False
    return True


def is_collage_possible(malign_mask_pth, normal_breast_pth):
    """
    Checks if collage is possible
    @param malign_mask_pth: Malign Mask path
    @param normal_breast_pth: Normal breast path
    @return: -1,-1 if collage is not possible | w,h if its possible
    """

    # Operations Threshold
    threshold = 50

    # Read the images
    malign_mask = cv2.imread(malign_mask_pth, cv2.IMREAD_GRAYSCALE)
    normal_breast = cv2.imread(normal_breast_pth, cv2.IMREAD_GRAYSCALE)
    normal_binary = image_to_binary(normal_breast, 'normal_aux.png')

    # Get images laterality
    R, _ = get_image_laterality(normal_breast)

    # Get images measures
    # Calculate malign mass measures
    m_top, m_right, m_bottom, m_left = get_measures(malign_mask)

    # Calculate normal breast measures
    n_top, n_right, n_bottom, n_left = get_measures(normal_binary)

    # Calculate widths and heights
    malign_mass_width = abs(m_right - m_left)
    malign_mass_height = abs(m_bottom - m_top)
    normal_breast_width = abs(n_right - n_left)
    normal_breast_height = abs(n_bottom - n_top)

    # Check if its worth the try
    if malign_mass_width > normal_breast_width or malign_mass_height > normal_breast_height:
        return -1, -1

    # Crop the malign mask
    crop_segmentation(malign_mask_pth, 'malign_aux.png')

    # Get bottom base coordinate
    base_coordinate = get_start_coordinate(normal_breast, 'malign_aux.png', R)

    # Coordinate collage starts bottom
    c, d = base_coordinate

    if R:

        # Go up until the masks match. If never match then skip them
        while c < normal_binary.shape[1]:
            if does_collage_mask(c, d, 'normal_aux.png'):
                return c, d

            c, d = c + threshold, d

        return -1, -1
    else:

        # Go up until the masks match. If never match then skip them
        while c >= threshold:
            if does_collage_mask(c, d, 'normal_aux.png'):
                return c, d

            c, d = c - threshold, d

        return -1, -1


# Remove the 4 channel to collage image
def remove_4_channel(im_path, output_path):
    """
    Remove the 4-channel tof the collage image
    @param im_path: Image to remove
    @param output_path: Output resulting image
    @return: -
    """
    imag = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    # Transpose naive image to properly see it
    transposed = imag.transpose(2, 0, 1)

    # Transpose image again with only the 3 rgb channels to save
    output = transposed[0:3].transpose(1, 2, 0)

    # Save new naive image (3-channels)
    cv2.imwrite(output_path, output)


# Resize image for hamronisation
def resize_image(im_path, percent_original, output_path):
    """
    Resizes image for harmonisation purposes
    @param im_path: Image path to resize
    @param percent_original: Percent of resize
    @param output_path: Output image path
    @return: -
    """
    imag = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ', imag.shape)

    scale_percent = percent_original  # percent of original size
    width = int(imag.shape[1] * scale_percent / 100)
    height = int(imag.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(imag, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)
    cv2.imwrite(output_path, resized)


def resize_to_dim(img_pth, width, height, out_pth):
    """
    Resizes a certain image to the specified resolution
    @param img_pth: Image to resize
    @param width: New width
    @param height: New height
    @param out_pth: Output path
    @return:
    """
    base = cv2.imread(img_pth, cv2.IMREAD_UNCHANGED)
    dim = (width, height)
    resized = cv2.resize(base, dim)

    if out_pth:
        cv2.imwrite(out_pth, resized)
        return True
    else:
        return resized


# Make mask have 3 channels
def make_3_channels_mask(im_path, out_path):
    """
    Make mask a three channel image
    @param im_path: Image path
    @param out_path: Output path
    @return: -
    """
    i = img.imread(im_path)
    new_i = [i, i, i]
    new_i = torch.tensor(np.array(new_i))
    tv.io.write_png(new_i, out_path)


# Crops the segmentation by its limits
def crop_segmentation(fp, outp):
    """
    Crops the image by its limits
    @param fp: Image filepath
    @param outp: Output path of cropped image
    @return: -
    """
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
    """
    Makes the collage
    @param malign_pth: Malign image path
    @param malign_mask_pth: Malign mask path
    @param normal_pth: Normal image path
    @param width: Width of collage point
    @param height: height of collage point
    @return: -
    """
    # Reads malign base image
    malign = cv2.imread(malign_pth, cv2.IMREAD_UNCHANGED)

    # Convert mask to 3 channels
    make_3_channels_mask(malign_mask_pth, 'malign_mask3.png')
    malign_mask = cv2.imread('malign_mask3.png', cv2.IMREAD_UNCHANGED)

    # Grab the image mask from the mass image
    masked = malign.copy()
    masked[malign_mask == 0] = 0
    cv2.imwrite('segmented_mass.png', masked)

    # Crop both the mask, and the masked mass
    crop_segmentation('segmented_mass.png', 'cropped_mass.png')
    crop_segmentation(malign_mask_pth, 'malign_mask_cropped.png')

    normal_image = Image.open(normal_pth)
    mass_to_paste = Image.open('cropped_mass.png')
    mass_mask = Image.open('malign_mask_cropped.png')

    # Creates collage and save
    back_im = normal_image.copy()
    back_im.paste(mass_to_paste, (width, height), mass_mask)
    back_im.save('collage.png', quality=95)

    # Creates collage mask
    collage_mask = Image.new("L", back_im.size, 0)
    collage_mask.paste(mass_mask, (width, height))
    collage_mask.save('collage_mask.png', quality=95)

    # Deletes unnecessary images
    try:
        os.remove('malign_mask3.png')
        os.remove('segmented_mass.png')
        os.remove('cropped_mass.png')
        os.remove('malign_mask_cropped.png')
    except OSError as e:
        print(f"FAILED\nFile: {e.filename}\nError: {e.strerror}")
