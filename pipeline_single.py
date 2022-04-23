# Required Imports
import os
import shutil
from argparse import ArgumentParser

import yaml

from utils.utils import get_image_core_name, execute_bash_command, get_latest_model, perform_collage

# Global Variables
GENERATED_FOLDER = "generated_images"
COLLAGES_FOLDER = "collage_images"
HARMONISED_FOLDER = "harmonised_images"
OPTIMISATION_FOLDER = "optimisation_images"
OPTIMISATION_BENIGN = "optimisation_images/benign"
OPTIMISATION_MALIGN = "optimisation_images/malign"
OUTPUT_FOLDER = "output_folder"
NORMAL_OUTPUT = "output_folder/normal"
BENIGN_OUTPUT = "output_folder/benign"
MALIGN_OUTPUT = "output_folder/malign"

# Variable to save the path of the main generators taking from reference path inside MedSinGAN
main_generators = {}


################################################
# GENERATION
################################################

def perform_generation(target_image, model_configurations):
    """

    @param target_image:
    @param model_configurations:
    @return:
    """

    # Change to correct directory
    os.chdir("MedSinGAN")

    # Get input core name
    core_name = get_image_core_name(target_image)

    # Train model on normal image
    command = f"python main_train.py --train_mode generation --input_name {target_image} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --convergence_patience {model_configurations['convergence_patience']} --batch_norm --gpu 0 "
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # Get the latest model
    latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")
    best_images_path = f"{latest_model}/gen_samples_stage_{model_configurations['stages'] - 1}"

    # Transfer generated images from last stage
    for generated_image in os.listdir(best_images_path):
        if not generated_image.startswith("."):
            shutil.move(os.path.join(best_images_path, generated_image), os.path.join("..", GENERATED_FOLDER))

    # Copy model info to later optimisation
    fixed_noise = f"{latest_model}/fixed_noise.pth"
    noise_amp = f"{latest_model}/noise_amp.pth"
    net_g = f"{latest_model}/{model_configurations['stages'] - 1}/netG.pth"
    net_d = f"{latest_model}/{model_configurations['stages'] - 1}/netD.pth"

    shutil.copy(fixed_noise, os.path.join("..", OPTIMISATION_BENIGN))
    shutil.copy(noise_amp, os.path.join("..", OPTIMISATION_BENIGN))
    shutil.copy(net_g, os.path.join("..", OPTIMISATION_BENIGN))
    shutil.copy(net_d, os.path.join("..", OPTIMISATION_BENIGN))

    shutil.copy(fixed_noise, os.path.join("..", OPTIMISATION_MALIGN))
    shutil.copy(noise_amp, os.path.join("..", OPTIMISATION_MALIGN))
    shutil.copy(net_g, os.path.join("..", OPTIMISATION_MALIGN))
    shutil.copy(net_d, os.path.join("..", OPTIMISATION_MALIGN))

    # Associate the normal folder to the main generators variable
    main_generators["normal"] = latest_model

    # Return back to the main folder
    os.chdir("../")


################################################
# COLLAGE
################################################

def perform_pipeline_collage(base_folder):
    """
    Performs collages with benign and malign images in the base folder, in the base images folder
    @param base_folder: Input folder with benign and malign mammography along segmentation
    @param base_images: Images to use as base for collages (usually normal generated images)
    @return: -
    """

    # Initiate the ids
    ids = 0

    # Perform collage with benign images
    benign_images = [os.path.join(base_folder, "benign", b_image) for b_image in
                     os.listdir(os.path.join(base_folder, "benign")) if
                     "_mask" not in b_image if not b_image.startswith(".")]

    # Perform collage with malign images
    malign_images = [os.path.join(base_folder, "malign", m_image) for m_image in
                     os.listdir(os.path.join(base_folder, "malign")) if
                     "_mask" not in m_image if not m_image.startswith(".")]

    for normal in os.listdir(GENERATED_FOLDER):

        for benign_image in benign_images:
            curr_normal = os.path.join(GENERATED_FOLDER, normal)
            curr_benign = benign_image
            curr_benign_mask = benign_image.replace(".png", "_mask.png")

            if perform_collage(curr_normal, curr_benign, curr_benign_mask) == 1:
                # Copy collage and mask to respective folder with respective name
                shutil.copy("collage.png", os.path.join(COLLAGES_FOLDER, f"benign_collage_{ids}.png"))
                shutil.copy("collage_mask.png", os.path.join(COLLAGES_FOLDER, f"benign_collage_{ids}_mask.png"))
                ids += 1

            else:
                print(f"Skipping collage between {normal} and {benign_image}!")

        for malign_image in malign_images:
            curr_normal = os.path.join(GENERATED_FOLDER, normal)
            curr_malign = malign_image
            curr_malign_mask = malign_image.replace(".png", "_mask.png")

            if perform_collage(curr_normal, curr_malign, curr_malign_mask) == 1:
                # Copy collage and mask to respective folder with respective name
                shutil.copy("collage.png", os.path.join(COLLAGES_FOLDER, f"malign_collage_{ids}.png"))
                shutil.copy("collage_mask.png", os.path.join(COLLAGES_FOLDER, f"malign_collage_{ids}_mask.png"))
                ids += 1

            else:
                print(f"Skipping collage between {normal} and {malign_image}!")

    # Remove the last collage and its mask
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
        cmd = f"python main_train.py --train_mode harmonization --gpu 0 --train_stages {model_configurations['stages']} --train_depth {model_configurations['concurrent']} --im_min_size {model_configurations['im_min_size']} --im_max_size {model_configurations['im_max_size']} --activation {model_configurations['act_func']} --lrelu_alpha {model_configurations['lrelu_alpha']} --niter {model_configurations['niter']} --convergence_patience {model_configurations['convergence_patience']} --batch_norm --input_name {os.path.join('..', MAIN_COLLAGE_FOLDER, folder, 'base_image.png')}"

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

            command = f"python main_train.py --train_mode generation --input_name {os.path.join('..', MAIN_HARMONISATION_FOLDER, folder, harmonised_image)} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --batch_norm --gpu 0 "
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
    # CREATE ALL REQUIRED FOLDERS ABOVE
    #######################################
    os.mkdir(GENERATED_FOLDER)
    os.mkdir(COLLAGES_FOLDER)
    os.mkdir(HARMONISED_FOLDER)
    os.mkdir(OPTIMISATION_FOLDER)
    os.mkdir(OPTIMISATION_BENIGN)
    os.mkdir(OPTIMISATION_MALIGN)
    os.mkdir(OUTPUT_FOLDER)
    os.mkdir(NORMAL_OUTPUT)
    os.mkdir(BENIGN_OUTPUT)
    os.mkdir(MALIGN_OUTPUT)

    #######################################
    # GENERATION
    #######################################

    # Perform generation training for each image type
    target_normal = os.listdir(os.path.join(opt_map.data_folder, "normal"))
    target_normal = [elem for elem in target_normal if not elem.startswith(".")]
    target_normal = os.path.join(opt_map.data_folder, target_normal[0])
    perform_generation(target_normal, configurations['generation'])

    #######################################
    # COLLAGE
    #######################################

    # Perform collages
    perform_pipeline_collage(opt_map.data_folder)

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
