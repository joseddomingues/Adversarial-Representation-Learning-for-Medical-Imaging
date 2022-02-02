# Imports
import os
from utils.utils import get_latest_model, make_collage
from argparse import ArgumentParser


def do_harmonisation_experiment(train_stages=3, min_size=120, lrelu_alpha=0.3, niter=1000,
                                base_img="../images/normal.png", naive_img="/content/collage.png",
                                experiment_name="(H)S3MS120L0.3N1000"):
    # Perform collage before using the images
    make_collage(malign_pth='/content/malign.png', malign_mask_pth='/content/malign_mask.png',
                 normal_pth='/content/normal.png', width=1000, height=1000)

    # Specify the variables

    # Run harmonisation train
    os.system(f"python main_train.py --train_mode harmonization --gpu 0 --train_stages {train_stages} "
              f"--im_min_size {min_size} --lrelu_alpha {lrelu_alpha} --niter {niter} --batch_norm --input_name {base_img} "
              f"--naive_img {naive_img} --experiment_name {experiment_name}")

    # Get the base image name
    core_name = base_img.split("/")[-1]
    core_name = core_name.split(".")[:-1]
    core_name = ".".join(core_name)

    curr = get_latest_model(f"/TrainedModels/{core_name}")

    # Fine tune
    if opt_map.fine_tune:
        os.system(
            f"python main_train.py --gpu 0 --train_mode harmonization --input_name {base_img} --naive_img {naive_img} "
            f"--fine_tune --model_dir " + str(curr))

    # Harmonise a given sample
    os.system(f"python evaluate_model.py --gpu 0 --model_dir {str(curr)} --naive_img {naive_img}")

    # Zip model and mlflows runs
    os.system(f"zip -r ../{HARMONISATION_MODELS_PATH}/{experiment_name}.zip .")

    # Delete current trained data
    os.system("rm -r /mlruns")
    os.system("rm -r /runs")
    os.system("rm -r /TrainedModels")


if __name__ == "__main__":

    arg = ArgumentParser()
    arg.add_argument('--fine_tune', dest='fine_tune', help='Flag to fine tune', action='store_true')
    arg.add_argument('--no_fine_tune', dest='fine_tune', help='Flag to not fine tune', action='store_false')
    arg.set_defaults(fine_tune=False)

    opt_map = arg.parse_args()

    # Check saving paths
    HARMONISATION_MODELS_PATH = "harmonisation_models"

    # Check if folder for models exists
    if not os.path.exists(HARMONISATION_MODELS_PATH):
        os.mkdir(HARMONISATION_MODELS_PATH)

    # Change to correct directory
    os.chdir("../MedSinGAN/")

    # Give image path
    image_name_path = "../images/normal.png"
    naive_im = "/content/collage.png"

    core_name = image_name_path.split("/")[-1]
    core_name = core_name.split(".")[:-1]
    core_name = ".".join(core_name)

    # Combinations to test
    stages = [12, 12, 16, 16, 20, 20]
    min_size = [5, 7, 7, 9, 9, 11]
    lrelu = [1500, 1500, 1500, 1500, 1500, 1500]
    niter = [1500, 1500, 1500, 1500, 1500, 1500]

    for comb in zip(stages, min_size, lrelu, niter):
        do_harmonisation_experiment(train_stages=comb[0], min_size=comb[1], lrelu_alpha=comb[2], niter=comb[3],
                                    base_img=image_name_path, naive_img=naive_im,
                                    experiment_name=f"(H)S{comb[0]}MS{comb[1]}L{comb[2]}N{comb[3]}_B{core_name}")
