# Required Imports
import os
import shutil
from argparse import ArgumentParser
from utils.utils import get_image_core_name, execute_bash_command, get_latest_model, is_collage_possible, make_collage, \
    make_3_channels_mask
import yaml

# Global Variables
MAIN_GENERATION_FOLDER = "generated_images"
MAIN_SEGMENTATION_FOLDER = "segmented_images"
MAIN_COLLAGE_FOLDER = "collage_images"
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
    # Get all data from benign folder
    images = os.listdir(os.path.join(base_folder, target_folder))
    images = [os.path.join("..", base_folder, target_folder, image) for image in images if "_mask" not in image]

    os.chdir("MedSinGAN")

    for image in images:
        core_name = get_image_core_name(image)
        current_folder = os.path.join("..", MAIN_GENERATION_FOLDER, target_folder, core_name)

        if not os.path.exists(current_folder):
            os.mkdir(current_folder)

        command = f"python main_train.py --train_mode generation --input_name {image} --train_stages {model_configurations['stages']} --niter {model_configurations['niter']} --train_depth {model_configurations['concurrent']} --gpu 0 "
        for path in execute_bash_command(command.split()):
            print(path, end="")

        latest_model = get_latest_model(base_path=f"TrainedModels/{core_name}")
        best_images_path = f"{latest_model}/gen_samples_stage_{model_configurations['stages'] - 1}"
        for generated_image in os.listdir(best_images_path):
            shutil.move(os.path.join(best_images_path, generated_image), current_folder)

        # Remove unnecessary folders from current generation
        command = "rm -r mlruns TrainedModels"
        for path in execute_bash_command(command.split()):
            print(path, end="")

    os.chdir("..")


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
# COLLAGE
################################################

def perform_collage(base_folder, base_images):
    """
    Performs collages with benign and malign images in the base folder, in the base images folder
    @param base_folder: Input folder with benign and malign mammography along segmentation
    @param base_images: Images to use as base for collages (usually normal generated images)
    @return: -
    """

    ids = 0

    for image_folder in os.listdir(base_images):

        normal_images = os.listdir(os.path.join(base_images, image_folder))
        for i in range(len(normal_images)):
            base_image = normal_images[i]

            # Create respective folder for current collage
            curr_collage_folder = os.path.join(MAIN_COLLAGE_FOLDER, str(i))
            os.mkdir(curr_collage_folder)

            # Copy base image to respective folder
            shutil.copy(os.path.join(base_images, base_image), os.path.join(curr_collage_folder, "base_image.png"))

            # Perform collage with benign images
            benign_images = [os.path.join(base_folder, DATA_FOLDER_BENIGN, b_image) for b_image in
                             os.listdir(os.path.join(base_folder, DATA_FOLDER_BENIGN)) if
                             "_mask" not in b_image]

            for benign_image in benign_images:
                w, h = is_collage_possible(benign_image.replace(".png", "_mask.png"),
                                           os.path.join(base_images, base_image))

                if w != -1 and h != -1:
                    make_collage(benign_image, benign_image.replace(".png", "_mask.png"),
                                 os.path.join(base_images, base_image), w, h)

                    # Make the collage mask 3-channel
                    make_3_channels_mask('collage_mask.png', 'collage_mask3.png')
                    os.remove('collage_mask.png')
                    os.rename('collage_mask3.png', 'collage_mask.png')

                    # Copy collage and mask to respective folder with respective name
                    shutil.copy("collage.png", os.path.join(curr_collage_folder, f"benign_collage_{ids}.png"))
                    shutil.copy("collage_mask.png", os.path.join(curr_collage_folder, f"benign_collage_{ids}_mask.png"))
                    ids += 1

                else:
                    print("Skipping collage!")

            # Perform collage with malign images
            malign_images = [os.path.join(base_folder, DATA_FOLDER_MALIGN, m_image) for m_image in
                             os.listdir(os.path.join(base_folder, DATA_FOLDER_MALIGN)) if
                             "_mask" not in m_image]

            for malign_image in malign_images:
                w, h = is_collage_possible(malign_image.replace(".png", "_mask.png"),
                                           os.path.join(base_images, base_image))

                if w != -1 and h != -1:
                    make_collage(malign_image, malign_image.replace(".png", "_mask.png"),
                                 os.path.join(base_images, base_image), w, h)

                    # Make the collage mask 3-channel
                    make_3_channels_mask('collage_mask.png', 'collage_mask3.png')
                    os.remove('collage_mask.png')
                    os.rename('collage_mask3.png', 'collage_mask.png')

                    # Copy collage and mask to respective folder with respective name
                    shutil.copy("collage.png", os.path.join(curr_collage_folder, f"malign_collage_{ids}.png"))
                    shutil.copy("collage_mask.png", os.path.join(curr_collage_folder, f"malign_collage_{ids}_mask.png"))
                    ids += 1

                else:
                    print("Skipping collage!")

    os.remove("collage_mask.png")
    os.remove("collage.png")
    os.remove("malign_aux.png")
    os.remove("normal_aux.png")


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

    for folder in os.listdir("..", MAIN_COLLAGE_FOLDER):

        # Create folder for harmonised images
        current_target = os.path.join("..", MAIN_HARMONISATION_FOLDER, folder)

        if not os.path.exists(current_target):
            os.mkdir(current_target)

        # Harmonise training with the current base image
        cmd = f"python main_train.py --train_mode harmonization --gpu 0 --train_stages {model_configurations['stages']} --im_min_size {model_configurations['im_min_size']} --lrelu_alpha {model_configurations['lrelu_alpha']} --niter {model_configurations['niter']} --batch_norm --input_name {os.path.join(MAIN_COLLAGE_FOLDER, folder, 'base_image.png')}"

        for path in execute_bash_command(cmd.split()):
            print(path, end="")

        # Harmonise the naive collage
        latest_model = get_latest_model(base_path="TrainedModels/base_image")
        collages = [col for col in os.listdir(os.path.join("..", MAIN_COLLAGE_FOLDER, folder)) if "collage" in col]
        collages = [col for col in collages if "_mask" not in col]

        for collage in collages:
            harmonise_cmd = "python evaluate_model.py --gpu 0 --model_dir " + str(
                latest_model) + " --naive_img " + os.path.join("..", MAIN_COLLAGE_FOLDER, folder, collage)

            for path in execute_bash_command(harmonise_cmd.split()):
                print(path, end="")

            target_harmonised = os.path.join(get_latest_model("TrainedModels/base_image"), "Evaluation_", "content",
                                             collage, "harmonized_w_mask.jpg")
            shutil.move(target_harmonised, os.path.join(current_target, collage.replace(".png", "_harmonised.png")))

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
    perform_collage(opt_map.data_folder, os.path.join(MAIN_GENERATION_FOLDER, DATA_FOLDER_NORMAL))

    #######################################
    # HARMONISATION
    #######################################

    # Check if folders exist or not. Create them if not
    if not os.path.exists(MAIN_HARMONISATION_FOLDER):
        os.mkdir(MAIN_HARMONISATION_FOLDER)

    # Performs harmonisation
    perform_harmonisation(opt_map.data_folder, configurations['harmonisation'])
