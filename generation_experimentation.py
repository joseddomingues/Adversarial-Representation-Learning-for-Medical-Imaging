# Imports
import os
from utils.utils import get_latest_model

# Check saving paths
GENERATORS_MODELS_PATH = "generator_models"
GENERATORS_MLFLOWS_PATH = "generator_flows"

if not os.path.exists(GENERATORS_MODELS_PATH):
    os.mkdir(GENERATORS_MODELS_PATH)

if not os.path.exists(GENERATORS_MLFLOWS_PATH):
    os.mkdir(GENERATORS_MLFLOWS_PATH)

# Change to correct directory
os.chdir("MedSinGAN/")

# Specify the variables
input_name = "../images/benign.png"
train_stages = "16"
train_depth = 9
n_iter = 2500
experiment_name = "(G)S6D9I2500"

# Get the base image name
core_name = input_name.split("/")[-1]
core_name = core_name.split(".")[:-1]
core_name = ".".join(core_name)

# Run generation train
os.system(
    f"python main_train.py --train_mode generation --input_name {input_name} --train_stages {train_stages} "
    f"--niter {n_iter} --train_depth {train_depth} --experiment_name {experiment_name} --gpu 0")

latest = get_latest_model(f"/TrainedModels/{core_name}/")

# Zip model and mlflows runs
os.system(f"zip -r ../{GENERATORS_MODELS_PATH}/{experiment_name}.zip {latest}")
os.system(f"zip -r ../{GENERATORS_MLFLOWS_PATH}/{experiment_name}.zip /mlruns")

# Delete current trained data
os.system("rm -r /mlruns")
os.system("rm -r /TrainedModels")
