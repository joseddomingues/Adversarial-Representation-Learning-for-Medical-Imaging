import os
from argparse import ArgumentParser

import cv2
import torch
from torch.autograd import Variable

# Creates argument parser
from MedSegmentation.metrics import dice_coeff, jaccard_index, accuracy_score

arg = ArgumentParser()

arg.add_argument('--model_dir', required=True, help='Checkpoints Model Directory Path')
arg.add_argument('--test_images', required=True, help='Test Images Directory Path')
arg.add_argument('--gt_images', required=True, help='Ground Truth Images Directory Path')
arg.add_argument('--output_folder', help='Output Folder To Save Segmentation Masks', default='results')

opt_map = arg.parse_args()

#######################################################
# Loads model from checkpoints
#######################################################
checkpoints_directory_unet = opt_map.model_dir
checkpoints_unet = os.listdir(checkpoints_directory_unet)
checkpoints_unet.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
model_unet = torch.load(os.path.join(checkpoints_directory_unet, checkpoints_unet[-1]))

# Check and create if needed the output folder
if not os.path.exists(opt_map.output_folder):
    os.mkdir(opt_map.output_folder)

#######################################################
# Checking for GPU
#######################################################
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

pin_memory = False
if train_on_gpu:
    pin_memory = True

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def _segment_image(image, model):
    """
    Apply segmentation on a given image with a given model
    @param image: Image to segment
    @param model: Model to use for segmentation
    @return: The segmentation for that image
    """
    model.eval()
    model.to(device)

    input_unet = cv2.imread(image)
    orig_width, orig_height = input_unet.shape[0], input_unet.shape[1]

    input_unet = input_unet.reshape((orig_width, orig_height, 3, 1))
    input_unet = input_unet.transpose((3, 2, 0, 1))

    input_unet.astype(float)
    input_unet = input_unet/255

    input_unet = torch.from_numpy(input_unet)
    input_unet = input_unet.type(torch.FloatTensor)
    input_unet = Variable(input_unet.to(device))

    out_unet = model(input_unet)
    out_unet = out_unet.cpu().data.numpy()
    out_unet = out_unet * 255

    out_unet = out_unet.transpose((2, 3, 0, 1))
    out_unet = out_unet.reshape((orig_width, orig_height, 1))
    return out_unet


#######################################################
# Go through each image in test images, segment it and save it
#######################################################
images = [file for file in os.listdir(opt_map.test_images) if not file.startswith('.')]
for image in images:
    curr_image = os.path.join(opt_map.test_images, image)
    result = _segment_image(curr_image, model_unet)
    name_aux = os.path.join(opt_map.output_folder, image.replace('.png', '_mask.png'))
    cv2.imwrite(name_aux, result)


#######################################################
# Evaluate segmented images
#######################################################
dice = 0
jacc = 0
acc = 0
pred_folder = opt_map.output_folder
or_folder = opt_map.gt_images
for i in os.listdir(or_folder):
  t = cv2.imread(os.path.join(or_folder, i), 0)
  p = cv2.imread(os.path.join(pred_folder, i), 0)

  t = t.reshape((t.shape[0], t.shape[1], 1)).transpose((2, 0, 1)).astype(float)/255
  p = p.reshape((p.shape[0], p.shape[1], 1)).transpose((2, 0, 1)).astype(float)/255

  dice += dice_coeff(t, p)
  jacc += jaccard_index(t, p)
  acc += accuracy_score(p, t)

print("Dice Coefficient:", dice/len(os.listdir(or_folder)))
print("Jaccard Index:", jacc/len(os.listdir(or_folder)))
print("Accuracy:", acc/len(os.listdir(or_folder)))
