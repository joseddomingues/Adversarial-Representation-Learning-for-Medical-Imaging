# Imports
import os
from utils.utils import execute_bash_command

# Check saving paths
SEGMENTATION_MODELS_PATH = "test_segmentations"


def do_segmentation_experiment(train_folder, val_folder, n_epochs, batch_size, experiment_name, model_checkpoints,
                               optimizer_checkpoints, l_rate, scheduler, test_images):
    """
    Run the segmentation experiment
    @param train_folder: Train folder
    @param val_folder: Validation folder
    @param n_epochs: N epochs
    @param batch_size: Batch size
    @param experiment_name: Experiment name
    @param model_checkpoints: Model checkpoints
    @param optimizer_checkpoints: Optimizer checkpoints
    @param l_rate: Learning rate
    @param scheduler: Scheduler type
    @param test_images: Test images if required
    @return: -
    """

    # Run segmentation train
    command = f"python train_Unet.py --train_folder {train_folder} --val_folder {val_folder} --n_epochs {n_epochs} --batch_size {batch_size} --experiment_name {experiment_name} --model_checkpoints {model_checkpoints} --optimizer_checkpoints {optimizer_checkpoints} --l_rate {l_rate} --scheduler {scheduler}"
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # Segment target folder
    command = f"python api.py --model_dir {model_checkpoints} --test_images {test_images} --no_eval"
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # Zip model and mlflows runs
    command = f"zip -r ../{SEGMENTATION_MODELS_PATH}/{experiment_name} ."
    for path in execute_bash_command(command.split()):
        print(path, end="")

    # Zip model and mlflows runs
    command = f"rm -r /mlruns /{model_checkpoints} /{optimizer_checkpoints} /runs /results"
    for path in execute_bash_command(command.split()):
        print(path, end="")


if __name__ == "__main__":

    # Check if folder for models exists
    if not os.path.exists(SEGMENTATION_MODELS_PATH):
        os.mkdir(SEGMENTATION_MODELS_PATH)

    # Change to correct directory
    os.chdir("../MedSegmentation/")

    # Combinations to test
    iters = [100, 200, 400, 600, 800, 1000]
    b_size = [1, 1, 1, 1, 1, 1]
    l_rate = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

    for comb in zip(iters, b_size, l_rate):
        exp_name = f"(S)I{comb[0]}B{comb[1]}LR{comb[2]}"

        do_segmentation_experiment(train_folder="../pipeline_data_inbreast", val_folder="../pipeline_data_inbreast", n_epochs=comb[0], batch_size=comb[1],
                                   experiment_name=exp_name, model_checkpoints=f"{exp_name}_model_checkpoints",
                                   optimizer_checkpoints=f"{exp_name}_optimizer_checkpoints", l_rate=comb[2],
                                   scheduler="cosine", test_images="/test_images")
