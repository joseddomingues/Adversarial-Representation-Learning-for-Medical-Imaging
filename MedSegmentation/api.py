import os
from argparse import ArgumentParser

import cv2
import torch
from torch.autograd import Variable

# Creates argument parser
arg = ArgumentParser()

arg.add_argument('--model_dir', required=True, help='Checkpoints Model Directory Path')
arg.add_argument('--test_images', required=True, help='Test Images Directory Path')
arg.add_argument('--output_folder', required=True, help='Output Folder To Save Segmentation Masks')

opt_map = arg.parse_args()

# Loads the model from the checkpoints
checkpoints_directory_unet = opt_map.model_dir  # "checkpoints_unet"
checkpoints_unet = os.listdir(checkpoints_directory_unet)
checkpoints_unet.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
model_unet = torch.load(checkpoints_directory_unet + '/' + checkpoints_unet[-1])

# Check and create if needed output folder
if not os.path.exists(opt_map.output_folder):
    os.mkdir(opt_map.output_folder)
else:
    os.remove(opt_map.output_folder)
    os.mkdir(opt_map.output_folder)


def _segment_image(image, model):
    """
    Apply segmentation on a given image with a given model
    @param image: Image to segment
    @param model: Model to use for segmentation
    @return: The segmentation for that image
    """
    model.eval()
    if torch.cuda.is_available():  # use gpu if available
        model.cuda()

    image = cv2.imread(image)
    orig_width, orig_height = image.shape[0], image.shape[1]
    input_unet = image

    input_unet = cv2.resize(input_unet, (256, 256), interpolation=cv2.INTER_CUBIC)
    input_unet = input_unet.reshape((256, 256, 3, 1))

    input_unet = input_unet.transpose((3, 2, 0, 1))

    input_unet.astype(float)
    input_unet = input_unet / 255

    input_unet = torch.from_numpy(input_unet)

    input_unet = input_unet.type(torch.FloatTensor)

    if torch.cuda.is_available():  # use gpu if available
        input_unet = Variable(input_unet.cuda())
    else:
        input_unet = Variable(input_unet)

    out_unet = model(input_unet)

    out_unet = out_unet.cpu().data.numpy()

    out_unet = out_unet * 255

    out_unet = out_unet.transpose((2, 3, 0, 1))
    out_unet = out_unet.reshape((256, 256, 1))
    out_unet = cv2.resize(out_unet, (orig_height, orig_width), interpolation=cv2.INTER_CUBIC)
    return out_unet


# For each image, segment the image and save it to the output folder
for image in os.listdir(opt_map.test_images):
    curr_image = opt_map.output_folder + '/' + image
    result = _segment_image(curr_image, model_unet)
    curr_image = curr_image.replace('.png', '_mask.png')
    cv2.imwrite(curr_image, result)
