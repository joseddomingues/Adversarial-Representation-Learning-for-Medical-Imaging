import json
import os

import cv2
import numpy as np

# https://towardsdatascience.com/generating-image-segmentation-masks-the-easy-way-dd4d3656dbd1
# https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html

source_folder = os.path.join(os.getcwd(), "images")
json_path = "mask_new_dataset.json"  # Relative to root directory
count = 0  # Count of total images saved
file_bbs = {}  # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 2457  # Dimensions should match those of ground truth image
MASK_HEIGHT = 1890

# Read JSON file
with open(json_path) as f:
    data = json.load(f)


# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
    except:
        print("No BB. Skipping", key)
        return

    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])

    file_bbs[key] = all_points


for itr in data:
    file_name_json = data[itr]["filename"]
    sub_count = 0  # Contains count of masks for a single ground truth image

    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count + 1)
            add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

print("\nDict size: ", len(file_bbs))

for file_name in os.listdir(source_folder):
    to_save_folder = os.path.join(source_folder, file_name[:-4])
    image_folder = os.path.join(to_save_folder, "images")
    mask_folder = os.path.join(to_save_folder, "masks")
    curr_img = os.path.join(source_folder, file_name)

    # make folders and copy image to new location
    os.mkdir(to_save_folder)
    os.mkdir(image_folder)
    os.mkdir(mask_folder)
    os.rename(curr_img, os.path.join(image_folder, file_name))

# For each entry in dictionary, generate mask and save in correponding
# folder


mask_folder = os.path.join(to_save_folder, "masks")
mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))

for itr in file_bbs:
    try:
        arr = np.array(file_bbs[itr])
    except:
        print("Not found:", itr)
        continue
    count += 1
    cv2.fillPoly(mask, [arr], color=(255))

cv2.imwrite(os.path.join(mask_folder, itr + ".png"), mask)

print("Images saved:", count)
