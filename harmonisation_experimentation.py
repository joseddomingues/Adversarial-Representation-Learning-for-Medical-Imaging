# Imports
import os
from utils.utils import get_latest_model, make_collage
from argparse import ArgumentParser

arg = ArgumentParser()
arg.add_argument('--fine_tune', dest='fine_tune', help='Flag to fine tune', action='store_true')
arg.add_argument('--no_fine_tune', dest='fine_tune', help='Flag to not fine tune', action='store_false')
arg.set_defaults(fine_tune=False)

opt_map = arg.parse_args()

# Check saving paths
HARMONISATION_MODELS_PATH = "harmonisation_models"
HARMONISATION_MLFLOWS_PATH = "harmonisation_flows"

if not os.path.exists(HARMONISATION_MODELS_PATH):
    os.mkdir(HARMONISATION_MODELS_PATH)

if not os.path.exists(HARMONISATION_MLFLOWS_PATH):
    os.mkdir(HARMONISATION_MLFLOWS_PATH)

# Change to correct directory
os.chdir("MedSinGAN/")

# Perform collage before using the images
make_collage(malign_pth='/content/malign.png', malign_mask_pth='/content/malign_mask.png', normal_pth='/content/normal.png', width=1000, height=1000)

# Specify the variables
train_stages = 3
min_size = 120
lrelu_alpha = 0.3
niter = 1000
base_img = "../images/normal.png"
naive_img = "/content/collage.png"
experiment_name = "(H)S3MS120L0.3N1000"

# Run harmonisation train
os.system(f"python main_train.py --train_mode harmonization --gpu 0 --train_stages {train_stages} "
          f"--im_min_size {min_size} --lrelu_alpha {lrelu_alpha} --niter {niter} --batch_norm --input_name {base_img} "
          f"--naive_img {naive_img} --experiment_name {experiment_name}")

curr = get_latest_model("/TrainedModels")

# Fine tune
if opt_map.fine_tune:
    os.system(
        f"python main_train.py --gpu 0 --train_mode harmonization --input_name {base_img} --naive_img {naive_img} "
        f"--fine_tune --model_dir " + str(curr))

# Zip model and mlflows runs
os.system(f"zip -r ../{HARMONISATION_MODELS_PATH}/{experiment_name}.zip {curr}")
os.system(f"zip -r ../{HARMONISATION_MLFLOWS_PATH}/{experiment_name}.zip /mlruns")

# Delete current trained data
os.system("rm -r /mlruns")
os.system("rm -r /TrainedModels")
