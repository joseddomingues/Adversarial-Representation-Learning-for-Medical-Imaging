import os
import shutil

input_folder = "/Users/josedaviddomingues/Desktop/harmonised_samples/16_stages_malign"
new_folder_name = "/Users/josedaviddomingues/Desktop/malign_harmonised"

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
