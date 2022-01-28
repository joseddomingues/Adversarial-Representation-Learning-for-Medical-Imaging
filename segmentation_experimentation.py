# Imports
import os

# Check saving paths
SEGMENTATION_MODELS_PATH = "test_segmentations"

if not os.path.exists(SEGMENTATION_MODELS_PATH):
    os.mkdir(SEGMENTATION_MODELS_PATH)

# Change to correct directory
os.chdir("MedSegmentation/")

# Specify the variables
train_folder = "../data"
val_folder = "../data"
n_epochs = 200
batch_size = 1
experiment_name = "(S)N100B1"
model_checkpoints = f"{experiment_name}_model_checkpoints"
optimizer_checkpoints = f"{experiment_name}_optimizer_checkpoints"
l_rate = 0.001
scheduler = "cosine"
test_images = "/test_images"

# Run segmentation train
os.system(
    f"python train_Unet.py --train_folder {train_folder} --val_folder {val_folder} --n_epochs {n_epochs} "
    f"--batch_size {batch_size} --experiment_name {experiment_name} --model_checkpoints {model_checkpoints} "
    f"--optimizer_checkpoints {optimizer_checkpoints} --l_rate {l_rate} --scheduler {scheduler}")

# Segment target folder
os.system(f"python api.py --model_dir {model_checkpoints} --test_images {test_images} --no_eval")

# Zip model and mlflows runs
os.system(f"zip -r ../{SEGMENTATION_MODELS_PATH}/{experiment_name} .")

# Zip model and mlflows runs
os.system("rm -r /mlruns")
os.system(f"rm -r /{model_checkpoints}")
os.system(f"rm -r /{optimizer_checkpoints}")
os.system(f"rm -r /runs")
os.system(f"rm -r /results")
