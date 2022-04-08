# Required Imports
import os
import shutil
from argparse import ArgumentParser

import yaml

from utils.utils import get_image_core_name, execute_bash_command, get_latest_model, perform_collage

# Global Variables
MAIN_GENERATION_FOLDER = "generated_images"
MAIN_SEGMENTATION_FOLDER = "segmented_images"
MAIN_COLLAGE_FOLDER = "collage_images"
MAIN_COLLAGE_GENERATION_FOLDER = "collage_generated_images"
MAIN_HARMONISATION_FOLDER = "harmonised_images"
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
    # Get all data from folder
    images = os.listdir(os.path.join(base_folder, target_folder))
    images = [os.path.join("..", base_folder, target_folder, image) for image in images if "_mask" not in image if
              not image.startswith(".")]

    os.chdir("MedSinGAN")

    for image in images:
        core_name = get_image_core_name(image)
        current_folder = os.path.join("..", MAIN_GENERATION_FOLDER, target_folder, core_name)

        if not os.path.exists(current_folder):
            os.mkdir(current_folder)

        command = f"python main_train.py --train_mode generation --input_name {image} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --im_min_size {model_configurations['im_min_size']} --batch_norm --gpu 0 "
        for path in execute_bash_command(command.split()):
            print(path, end="")

        latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")
        best_images_path = f"{latest_model}/gen_samples_stage_{model_configurations['stages'] - 1}"
        for generated_image in os.listdir(best_images_path):
            if not generated_image.startswith("."):
                shutil.move(os.path.join(best_images_path, generated_image), current_folder)

        # Remove unnecessary folders from current generation
        command = "rm -r mlruns TrainedModels"
        for path in execute_bash_command(command.split()):
            print(path, end="")

    os.chdir("../")


################################################
# COLLAGE
################################################

def perform_pipeline_collage(base_folder, base_images):
    """
    Performs collages with benign and malign images in the base folder, in the base images folder
    @param base_folder: Input folder with benign and malign mammography along segmentation
    @param base_images: Images to use as base for collages (usually normal generated images)
    @return: -
    """

    for image_folder in os.listdir(base_images):

        ids = 0
        normal_images = os.listdir(os.path.join(base_images, image_folder))
        normal_images = [image for image in normal_images if not image.startswith(".")]

        for i in range(len(normal_images)):
            base_image = normal_images[i]

            # Create respective folder for current collage
            curr_collage_folder = os.path.join(MAIN_COLLAGE_FOLDER, f"{image_folder}_{i}")
            os.mkdir(curr_collage_folder)

            # Copy base image to respective folder
            shutil.copy(os.path.join(base_images, image_folder, base_image),
                        os.path.join(curr_collage_folder, "base_image.png"))

            # Perform collage with benign images
            benign_images = [os.path.join(base_folder, DATA_FOLDER_BENIGN, b_image) for b_image in
                             os.listdir(os.path.join(base_folder, DATA_FOLDER_BENIGN)) if
                             "_mask" not in b_image if not b_image.startswith(".")]

            for benign_image in benign_images:
                curr_normal = os.path.join(base_images, image_folder, base_image)
                curr_benign = benign_image
                curr_benign_mask = benign_image.replace(".png", "_mask.png")

                if perform_collage(curr_normal, curr_benign, curr_benign_mask) == 1:
                    # Copy collage and mask to respective folder with respective name
                    shutil.copy("collage.png", os.path.join(curr_collage_folder, f"benign_collage_{ids}.png"))
                    shutil.copy("collage_mask.png", os.path.join(curr_collage_folder, f"benign_collage_{ids}_mask.png"))
                    ids += 1

                else:
                    print("Skipping collage!")

            # Perform collage with malign images
            malign_images = [os.path.join(base_folder, DATA_FOLDER_MALIGN, m_image) for m_image in
                             os.listdir(os.path.join(base_folder, DATA_FOLDER_MALIGN)) if
                             "_mask" not in m_image if not m_image.startswith(".")]

            for malign_image in malign_images:
                curr_normal = os.path.join(base_images, image_folder, base_image)
                curr_malign = malign_image
                curr_malign_mask = malign_image.replace(".png", "_mask.png")

                if perform_collage(curr_normal, curr_malign, curr_malign_mask) == 1:
                    # Copy collage and mask to respective folder with respective name
                    shutil.copy("collage.png", os.path.join(curr_collage_folder, f"malign_collage_{ids}.png"))
                    shutil.copy("collage_mask.png", os.path.join(curr_collage_folder, f"malign_collage_{ids}_mask.png"))
                    ids += 1

                else:
                    print("Skipping collage!")

    os.remove("collage_mask.png")
    os.remove("collage.png")


################################################
# HARMONISATION
################################################

def perform_harmonisation(model_configurations):
    """
    Performs the harmonisation process on the made collages
    @param model_configurations: Harmoniser model configurations
    @return: -
    """

    # Change to the correct directory
    os.chdir("MedSinGAN")

    for folder in os.listdir(os.path.join("..", MAIN_COLLAGE_FOLDER)):

        # Create folder for harmonised images
        current_target = os.path.join("..", MAIN_HARMONISATION_FOLDER, folder)

        if not os.path.exists(current_target):
            os.mkdir(current_target)

        # Harmonise training with the current base image
        cmd = f"python main_train.py --train_mode harmonization --gpu 0 --train_stages {model_configurations['stages']} --train_depth {model_configurations['concurrent']} --im_min_size {model_configurations['im_min_size']} --im_max_size {model_configurations['im_max_size']} --activation {model_configurations['act_func']} --lrelu_alpha {model_configurations['lrelu_alpha']} --niter {model_configurations['niter']} --batch_norm --input_name {os.path.join('..', MAIN_COLLAGE_FOLDER, folder, 'base_image.png')}"

        for path in execute_bash_command(cmd.split()):
            print(path, end="")

        # Harmonise the naive collage
        latest_model = get_latest_model(base_path="TrainedModels/base_image")
        collages = [col for col in os.listdir(os.path.join("..", MAIN_COLLAGE_FOLDER, folder)) if "collage" in col if
                    "_mask" not in col if not col.startswith(".")]

        for collage in collages:
            harmonise_cmd = "python evaluate_model.py --gpu 0 --model_dir " + str(
                latest_model) + " --naive_img " + os.path.join("..", MAIN_COLLAGE_FOLDER, folder, collage)

            for path in execute_bash_command(harmonise_cmd.split()):
                print(path, end="")

            target_harmonised = os.path.join(get_latest_model("TrainedModels/base_image"), "Evaluation_..",
                                             MAIN_COLLAGE_FOLDER, folder,
                                             collage, "harmonized_w_mask.jpg")
            shutil.move(target_harmonised, os.path.join(current_target, collage.replace(".png", "_harmonised.png")))

        # Remove unnecessary folders from current generation
        command = "rm -r mlruns TrainedModels"
        for path in execute_bash_command(command.split()):
            print(path, end="")

    os.chdir("..")


def do_collage_generation(model_configurations):
    """
    Perform the generation process over the collages made previously
    @param model_configurations: Model configurations for the generator
    @return: - 
    """

    for folder in os.listdir(MAIN_HARMONISATION_FOLDER):

        for harmonised_image in os.listdir(os.path.join(MAIN_HARMONISATION_FOLDER, folder)):

            if harmonised_image.startswith("."):
                continue

            curr_target = os.path.join(MAIN_COLLAGE_GENERATION_FOLDER, folder)
            if not os.path.exists(curr_target):
                os.mkdir(curr_target)

            curr_target = os.path.join(curr_target, harmonised_image.split(".")[0])
            if not os.path.exists(curr_target):
                os.mkdir(curr_target)

            os.chdir("MedSinGAN")

            command = f"python main_train.py --train_mode generation --input_name {os.path.join('..', MAIN_HARMONISATION_FOLDER, folder, harmonised_image)} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --im_min_size {model_configurations['im_min_size']} --batch_norm --gpu 0 "
            for path in execute_bash_command(command.split()):
                print(path, end="")

            core_name = get_image_core_name(harmonised_image)
            latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")
            best_images_path = f"{latest_model}/gen_samples_stage_{model_configurations['stages'] - 1}"
            for generated_image in os.listdir(best_images_path):
                if not generated_image.startswith("."):
                    shutil.move(os.path.join(best_images_path, generated_image), os.path.join("..", curr_target))

            # Remove unnecessary folders from current generation
            command = "rm -r mlruns TrainedModels"
            for path in execute_bash_command(command.split()):
                print(path, end="")

            os.chdir("..")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_folder", type=str, required=True, help="Folder where data is. Organisation sensitive.")
    args.add_argument("--pipeline_config", type=str, required=True,
                      help="Pipeline Configurations YAML file. Organisation Sensitive")

    opt_map = args.parse_args()

    #######################################
    # YAML FILE PROCESSING
    #######################################

    with open(opt_map.pipeline_config, "r") as stream:
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

    if not os.path.exists(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL)):
        os.mkdir(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL))

    # Perform generation training for each image type
    perform_generation(opt_map.data_folder, DATA_FOLDER_NORMAL, configurations['generation'])

    #######################################
    # COLLAGE
    #######################################

    # Check if folders exist or not. Create them if not
    if not os.path.exists(MAIN_COLLAGE_FOLDER):
        os.mkdir(MAIN_COLLAGE_FOLDER)

    # Perform collages
    perform_pipeline_collage(opt_map.data_folder, os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL))

    #######################################
    # HARMONISATION
    #######################################

    # Check if folders exist or not. Create them if not
    if not os.path.exists(MAIN_HARMONISATION_FOLDER):
        os.mkdir(MAIN_HARMONISATION_FOLDER)

    # Performs harmonisation
    perform_harmonisation(configurations['harmonisation'])

    #######################################
    # GENERATION - II
    #######################################

    # Check if folders exist or not. Create them if not
    if not os.path.exists(MAIN_COLLAGE_GENERATION_FOLDER):
        os.mkdir(MAIN_COLLAGE_GENERATION_FOLDER)

    if not os.path.exists(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL)):
        os.mkdir(os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL))

    # Perform generation training for each image type
    do_collage_generation(configurations['generation'])
