# Required Imports
import os
import shutil
from argparse import ArgumentParser
from utils.utils import get_image_core_name, execute_bash_command, get_latest_model
import yaml

# Global Variables
MAIN_GENERATION_FOLDER = "generated_images"
MAIN_SEGMENTATION_FOLDER = "segmented_images"
DATA_FOLDER_BENIGN = "benign"
DATA_FOLDER_MALIGN = "malign"
DATA_FOLDER_NORMAL = "normal"


################################################
# GENERATION
################################################

def perform_generation(base_folder, target_folder, model_configurations):
    """
    Performs generation task on the given input folders
    @param base_folder: Base folder for base images
    @param target_folder: Target folder to save images
    @param model_configurations: Model configurations to use in the pipeline
    @return: -
    """
    # Get all data from benign folder
    images = os.listdir(os.path.join(base_folder, target_folder))
    images = [os.path.join("..", base_folder, target_folder, image) for image in images if "_mask" not in image]

    os.chdir('MedSinGAN')

    for image in images:
        core_name = get_image_core_name(image)
        current_folder = os.path.join(MAIN_GENERATION_FOLDER, target_folder, core_name)
        os.mkdir(current_folder)

        command = f"python main_train.py --train_mode generation --input_name {image} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --gpu 0 "
        for path in execute_bash_command(command.split()):
            print(path, end="")

        latest_model = get_latest_model(base_path=f"/TrainedModels/{core_name}")
        best_images_path = f"{latest_model}/gen_samples_stage_{model_configurations['stages'] - 1}"
        for generated_image in os.listdir(best_images_path):
            shutil.move(os.path.join(best_images_path, generated_image), os.path.join('..', current_folder))


################################################
# SEGMENTATION
################################################

def perform_segmentation(base_folder, model_configurations, target_benign_segmentations_folder,
                         target_malign_segmentations_folder):
    """
    Performs the segmentation task over the benign and malign images
    using the available masks to train the segmentation network
    @param base_folder: Main base folder where data is
    @param model_configurations: Segmentation model configurations
    @param target_benign_segmentations_folder: Target folder to save benign segmentations
    @param target_malign_segmentations_folder: Target folder to save malign segmentations
    @return: -
    """
    # Get the images and respective segmentation_masks, along with the non mask images
    # Benign
    benign_masks = os.listdir(os.path.join(base_folder, DATA_FOLDER_BENIGN))
    benign_masks = [os.path.join("..", base_folder, DATA_FOLDER_BENIGN, elem) for elem in benign_masks if
                    "_mask" in elem]
    benign_images = [''.join(elem.split("_mask")) for elem in benign_masks]

    non_mask_benign = [os.path.join("..", base_folder, DATA_FOLDER_BENIGN, elem) for elem in
                       (os.path.join('..', base_folder, DATA_FOLDER_BENIGN)) if "_mask" not in elem]
    non_mask_benign = [elem for elem in non_mask_benign if elem not in benign_images]

    # Malign
    malign_masks = os.listdir(os.path.join(base_folder, DATA_FOLDER_MALIGN))
    malign_masks = [elem for elem in malign_masks if "_mask" in elem]
    malign_images = [''.join(elem.split("_mask")) for elem in malign_masks]

    non_mask_malign = [os.path.join("..", base_folder, DATA_FOLDER_MALIGN, elem) for elem in
                       (os.path.join('..', base_folder, DATA_FOLDER_MALIGN)) if "_mask" not in elem]
    non_mask_malign = [elem for elem in non_mask_malign if elem not in malign_images]

    # Change to model directory
    os.chdir('MedSegmentation')

    # Create temp folder
    tmp_folder = "/temp_train"
    os.mkdir(tmp_folder)

    # Create a copy of the files into the temp train folder
    for elem in benign_images:
        shutil.copy(elem, tmp_folder)

    for elem in benign_masks:
        shutil.copy(elem, tmp_folder)

    for elem in malign_images:
        shutil.copy(elem, tmp_folder)

    for elem in malign_masks:
        shutil.copy(elem, tmp_folder)

    # Trains unet with data
    command = f"python train_Unet.py --train_folder {tmp_folder} --val_folder {tmp_folder} --n_epochs {model_configurations['niter']} --batch_size {model_configurations['b_size']} --l_rate {model_configurations['l_rate']} --scheduler {model_configurations['scheduler']}"
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # Removes temp folder
    command = f"rm -r {tmp_folder}"
    execute_bash_command(command.split())

    # Segment the images from the benign folder
    command = f"python api.py --model_dir /model_checkpoints --test_images {os.path.join('..', base_folder, DATA_FOLDER_BENIGN)} --no_eval --output_folder {os.path.join('..', target_benign_segmentations_folder)}"
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # Segment the images from the malign folder
    command = f"python api.py --model_dir /model_checkpoints --test_images {os.path.join('..', base_folder, DATA_FOLDER_MALIGN)} --no_eval --output_folder {os.path.join('..', target_malign_segmentations_folder)}"
    for path in execute_bash_command(command.split()):
        print(path, end="")


################################################
# HARMONISATION
################################################

def perform_harmonisation():
    print("TOD BE DONE....")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_folder", type=str, required=True, help="Folder where data is. Organisation sensitive.")
    args.add_argument("--pipeline_config", type=str, required=True,
                      help="Pipeline Configurations YAML file. Organisation .Sensitive")

    opt_map = args.parse_args()

    #######################################
    # YAML FILE PROCESSING
    #######################################

    with open(opt_map['pipeline_config'], "r") as stream:
        try:
            configurations = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

    #######################################
    # GENERATION
    #######################################

    # Check if folders exist or not. Create them if not
    if not os.path.exists(MAIN_GENERATION_FOLDER):
        os.mkdir(MAIN_GENERATION_FOLDER)

    if not os.path.exists(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_BENIGN)):
        os.mkdir(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_BENIGN))

    if not os.path.exists(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_MALIGN)):
        os.mkdir(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_MALIGN))

    if not os.path.exists(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL)):
        os.mkdir(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL))

    # Perform generation training for each image type
    perform_generation(opt_map['data_folder'], DATA_FOLDER_BENIGN, configurations['generation'])
    perform_generation(opt_map['data_folder'], DATA_FOLDER_MALIGN, configurations['generation'])
    perform_generation(opt_map['data_folder'], DATA_FOLDER_NORMAL, configurations['generation'])

    #######################################
    # SEGMENTATION
    #######################################

    # Check if folders exist or not. If not then create them
    # Not required for normal images since they don't have masses
    if not os.path.exists(MAIN_SEGMENTATION_FOLDER):
        os.mkdir(MAIN_SEGMENTATION_FOLDER)

    if not os.path.exists(os.path.join(MAIN_SEGMENTATION_FOLDER, DATA_FOLDER_BENIGN)):
        os.mkdir(os.path.join(MAIN_SEGMENTATION_FOLDER, DATA_FOLDER_BENIGN))

    if not os.path.exists(os.path.join(MAIN_SEGMENTATION_FOLDER, DATA_FOLDER_MALIGN)):
        os.mkdir(os.path.join(MAIN_SEGMENTATION_FOLDER, DATA_FOLDER_MALIGN))

    # Perform segmentation training, and segment images
    perform_segmentation(opt_map['data_folder'], configurations['segmentation'],
                         os.path.join(MAIN_SEGMENTATION_FOLDER, DATA_FOLDER_BENIGN),
                         os.path.join(MAIN_SEGMENTATION_FOLDER, DATA_FOLDER_MALIGN))
