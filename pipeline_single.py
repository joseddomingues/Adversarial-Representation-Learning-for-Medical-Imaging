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
    training_image = os.path.join("..", target_image)
    core_name = get_image_core_name(target_image)

    # Train model on normal image
    command = f"python main_train.py --train_mode generation --input_name {training_image} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --convergence_patience {model_configurations['convergence_patience']} --batch_norm --gpu 0 "
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

def perform_pipeline_collage(base_folder, base_image):
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
            curr_benign = benign_image
            curr_benign_mask = benign_image.replace(".png", "_mask.png")

            if perform_collage(base_image, curr_benign, curr_benign_mask) == 1:
                # Copy collage and mask to respective folder with respective name
                shutil.move("collage.png", os.path.join(COLLAGES_FOLDER, f"benign_collage_{ids}.png"))
                shutil.move("collage_mask.png", os.path.join(COLLAGES_FOLDER, f"benign_collage_{ids}_mask.png"))
                ids += 1

            else:
                print(f"Skipping collage between {normal} and {benign_image}!")

        for malign_image in malign_images:
            curr_malign = malign_image
            curr_malign_mask = malign_image.replace(".png", "_mask.png")

            if perform_collage(base_image, curr_malign, curr_malign_mask) == 1:
                # Copy collage and mask to respective folder with respective name
                shutil.move("collage.png", os.path.join(COLLAGES_FOLDER, f"malign_collage_{ids}.png"))
                shutil.move("collage_mask.png", os.path.join(COLLAGES_FOLDER, f"malign_collage_{ids}_mask.png"))
                ids += 1

            else:
                print(f"Skipping collage between {normal} and {malign_image}!")


################################################
# HARMONISATION
################################################

def perform_harmonisation(base_image, model_configurations):
    """
    Performs the harmonisation process on the made collages
    @param model_configurations: Harmoniser model configurations
    @return: -
    """

    # Change to the correct directory
    os.chdir("MedSinGAN")

    # Harmonise training with the current base image
    cmd = f"python main_train.py --train_mode harmonization --gpu 0 --train_stages {model_configurations['stages']} --train_depth {model_configurations['concurrent']} --im_min_size {model_configurations['im_min_size']} --im_max_size {model_configurations['im_max_size']} --activation {model_configurations['act_func']} --lrelu_alpha {model_configurations['lrelu_alpha']} --niter {model_configurations['niter']} --convergence_patience {model_configurations['convergence_patience']} --batch_norm --input_name {os.path.join('..', base_image)}"
    for path in execute_bash_command(cmd.split()):
        print(path, end="")

    # Harmonise the naive collage
    core_name = get_image_core_name(base_image)
    latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")

    collages = [col for col in os.listdir(os.path.join("..", COLLAGES_FOLDER)) if "collage" in col if
                "_mask" not in col if not col.startswith(".")]

    for collage in collages:
        harmonise_cmd = f"python evaluate_model.py --gpu 0 --model_dir {latest_model} --naive_img {os.path.join('..', COLLAGES_FOLDER, collage)}"

        for path in execute_bash_command(harmonise_cmd.split()):
            print(path, end="")

        target_harmonised = os.path.join(latest_model, "Evaluation_..", COLLAGES_FOLDER, collage,
                                         "harmonized_w_mask.jpg")
        shutil.move(target_harmonised,
                    os.path.join("..", HARMONISED_FOLDER, collage.replace(".png", "_harmonised.png")))

    os.chdir("..")


def perform_optimisation(model_configurations, target_image):
    """

    @param model_configurations:
    @return:
    """

    # Copy files to respective folders
    benign_harmonised = [os.path.join(HARMONISED_FOLDER, elem) for elem in os.listdir(HARMONISED_FOLDER) if
                         "benign" in elem if not elem.startswith(".")]
    malign_harmonised = [os.path.join(HARMONISED_FOLDER, elem) for elem in os.listdir(HARMONISED_FOLDER) if
                         "malign" in elem if not elem.startswith(".")]

    benign_opt_stages = model_configurations['stages']
    malign_opt_stages = model_configurations['stages']

    for benign in benign_harmonised:
        shutil.copy(benign, OPTIMISATION_BENIGN)

    for malign in malign_harmonised:
        shutil.copy(malign, OPTIMISATION_MALIGN)

    os.chdir("MedSinGAN")

    # Optimise for the benign images
    folder_benign = os.path.join("..", OPTIMISATION_BENIGN)
    benign_opt_ims = [elem for elem in os.listdir(folder_benign) if ".pth" not in elem if not elem.startswith(".")]
    for ben in os.listdir(benign_opt_ims):

        command = f"python main_train.py --train_mode generation --input_name {os.path.join(folder_benign, ben)} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {benign_opt_stages} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --batch_norm --convergence_patience {model_configurations['convergence_patience']} --g_optimizer_folder {os.path.join('..', OPTIMISATION_BENIGN)} --gpu 0 "
        for path in execute_bash_command(command.split()):
            print(path, end="")

        core_name = get_image_core_name(target_image)
        latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")

        # Update stages
        benign_opt_stages += 1

        # Copy current files to the optimisation folder
        fixed_noise = f"{latest_model}/fixed_noise.pth"
        noise_amp = f"{latest_model}/noise_amp.pth"
        net_g = f"{latest_model}/{benign_opt_stages}/netG.pth"
        net_d = f"{latest_model}/{benign_opt_stages}/netD.pth"

        shutil.copy(fixed_noise, folder_benign)
        shutil.copy(noise_amp, folder_benign)
        shutil.copy(net_g, folder_benign)
        shutil.copy(net_d, folder_benign)

        # Associate the benign folder to the main generators variable
        main_generators["benign"] = latest_model

    # Optimise for the malign images
    folder_malign = os.path.join("..", OPTIMISATION_MALIGN)
    malign_opt_ims = [elem for elem in os.listdir(folder_malign) if ".pth" not in elem if not elem.startswith(".")]
    for mal in os.listdir(malign_opt_ims):

        command = f"python main_train.py --train_mode generation --input_name {os.path.join(folder_malign, mal)} --n_samples_generate {model_configurations['n_samples_generate']} --train_stages {malign_opt_stages} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --activation {model_configurations['act_func']} --im_max_size {model_configurations['im_max_size']} --batch_norm --convergence_patience {model_configurations['convergence_patience']} --g_optimizer_folder {os.path.join('..', OPTIMISATION_MALIGN)} --gpu 0 "
        for path in execute_bash_command(command.split()):
            print(path, end="")

        core_name = get_image_core_name(target_image)
        latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")

        # Update stages
        malign_opt_stages += 1

        # Copy current files to the optimisation folder
        fixed_noise = f"{latest_model}/fixed_noise.pth"
        noise_amp = f"{latest_model}/noise_amp.pth"
        net_g = f"{latest_model}/{malign_opt_stages}/netG.pth"
        net_d = f"{latest_model}/{malign_opt_stages}/netD.pth"

        shutil.copy(fixed_noise, folder_malign)
        shutil.copy(noise_amp, folder_malign)
        shutil.copy(net_g, folder_malign)
        shutil.copy(net_d, folder_malign)

        # Associate the benign folder to the main generators variable
        main_generators["malign"] = latest_model

    os.chdir("..")


def final_generation(samples_to_generate):
    os.chdir("MedSinGAN")
    samples_dir = "Evaluation_/random_samples"

    # Generate benign samples
    command = f"python evaluate_model.py --gpu 0 --model_dir {main_generators['benign']} --num_samples {samples_to_generate}"
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # move images to correct folder
    for sample in os.listdir(os.path.join(main_generators['benign'], samples_dir)):
        if not sample.startswith("."):
            shutil.move(os.path.join(main_generators['benign'], samples_dir, sample), os.path.join("..", BENIGN_OUTPUT))

    # Generate malign samples
    command = f"python evaluate_model.py --gpu 0 --model_dir {main_generators['malign']} --num_samples {samples_to_generate}"
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # move images to correct folder
    for sample in os.listdir(os.path.join(main_generators['malign'], samples_dir)):
        if not sample.startswith("."):
            shutil.move(os.path.join(main_generators['malign'], samples_dir, sample), os.path.join("..", MALIGN_OUTPUT))

    # Generate normal samples
    command = f"python evaluate_model.py --gpu 0 --model_dir {main_generators['normal']} --num_samples {samples_to_generate}"
    for path in execute_bash_command(command.split()):
        print(path, end="")

        # move images to correct folder
        for sample in os.listdir(os.path.join(main_generators['normal'], samples_dir)):
            if not sample.startswith("."):
                shutil.move(os.path.join(main_generators['normal'], samples_dir, sample),
                            os.path.join("..", NORMAL_OUTPUT))


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_folder", type=str, required=True, help="Folder where data is. Organisation sensitive.")
    args.add_argument("--samples_for_output", type=int, required=True,
                      help="Number of samples for each mammography type")
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
    target_normal = os.path.join(opt_map.data_folder, "normal", target_normal[0])
    perform_generation(target_normal, configurations['generation'])

    #######################################
    # COLLAGE
    #######################################

    perform_pipeline_collage(opt_map.data_folder, target_normal)

    #######################################
    # HARMONISATION
    #######################################

    perform_harmonisation(target_normal, configurations['harmonisation'])

    #######################################
    # OPTIMISATION
    #######################################

    perform_optimisation(configurations['generation'], target_normal)

    #######################################
    # FINAL GENERATION
    #######################################

    final_generation(opt_map.samples_for_output)
