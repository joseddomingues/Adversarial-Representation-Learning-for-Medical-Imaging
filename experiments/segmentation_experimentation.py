# Imports
import os

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
        exp_name = f"(S)N{comb[0]}B{comb[1]}LR{comb[2]}"

        do_segmentation_experiment(train_folder="../data", val_folder="../data", n_epochs=comb[0], batch_size=comb[1],
                                   experiment_name=exp_name, model_checkpoints=f"{exp_name}_model_checkpoints",
                                   optimizer_checkpoints=f"{exp_name}_optimizer_checkpoints", l_rate=comb[2],
                                   scheduler="cosine", test_images="/test_images")
