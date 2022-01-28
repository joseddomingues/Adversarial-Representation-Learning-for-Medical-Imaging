# Imports
import os

# Check saving paths
GENERATORS_MODELS_PATH = "test_generators"

if not os.path.exists(GENERATORS_MODELS_PATH):
    os.mkdir(GENERATORS_MODELS_PATH)

# Change to correct directory
os.chdir("MedSinGAN/")

# Specify the variables
input_name = "../images/benign.png"
train_stages = "16"
train_depth = 9
n_iter = 2500
experiment_name = "(G)S6D9I2500"

# Run generation train
os.system(
    f"python main_train.py --train_mode generation --input_name {input_name} --train_stages {train_stages} "
    f"--niter {n_iter} --train_depth {train_depth} --experiment_name {experiment_name} --gpu 0")

# Zip model and mlflows runs
os.system(f"zip -r ../{GENERATORS_MODELS_PATH}/{experiment_name}.zip .")

# Delete current trained data
os.system("rm -r /TrainedModels")
os.system("rm -r /mlruns")
os.system("rm -r /runs")
