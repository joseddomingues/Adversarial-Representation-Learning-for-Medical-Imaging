# Required Imports
import os
import shutil
from argparse import ArgumentParser
from utils.utils import get_image_core_name, execute_bash_command, get_latest_model
import yaml

# Global Variables
MAIN_GENERATION_FOLDER = "generated_images"
DATA_FOLDER_BENIGN = "benign"
DATA_FOLDER_MALIGN = "malign"
DATA_FOLDER_NORMAL = "normal"


################################################
# GENERATION
################################################

def perform_generation(base_folder, target_folder, model_configurations):
    # Get all data from benign folder
    images = os.listdir(os.path.join(base_folder, target_folder))
    images = [os.path.join(base_folder, target_folder, image) for image in images if "_mask" not in image]

    for image in images:
        core_name = get_image_core_name(image)
        current_folder = os.path.join(MAIN_GENERATION_FOLDER, target_folder, core_name)
        os.mkdir(current_folder)

        os.chdir('MedSinGAN')
        command = f"python main_train.py --train_mode generation --input_name {image} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --gpu 0 "
        for path in execute_bash_command(command.split()):
            print(path, end="")

        latest_model = get_latest_model(base_path="/TrainedModels/{core_name}/")
        best_images_path = f"/TrainedModels/{core_name}/{latest_model}/gen_samples_stage_{model_configurations['stages'] - 1}"
        for generated_image in os.listdir(best_images_path):
            shutil.move(os.path.join(best_images_path, generated_image), os.path.join('..', current_folder))


################################################
# SEGMENTATION
################################################

################################################
# HARMONISATION
################################################


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
    print()
