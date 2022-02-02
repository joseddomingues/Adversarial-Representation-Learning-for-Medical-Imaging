# Imports
import os

# Check saving paths
GENERATORS_MODELS_PATH = "test_generators"


def do_generation_experiment(input_name, train_stages, train_depth, n_iter, experiment_name):
    """
    Run the generation experiment
    @param input_name: Input name
    @param train_stages: Train stages
    @param train_depth: Train depth
    @param n_iter: N iterations
    @param experiment_name: Experiment name
    @return: -
    """

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


if __name__ == "__main__":

    # Check if folder for models exists
    if not os.path.exists(GENERATORS_MODELS_PATH):
        os.mkdir(GENERATORS_MODELS_PATH)

    # Change to correct directory
    os.chdir("MedSinGAN/")

    do_generation_experiment(input_name="../images/benign.png", train_stages=16, train_depth=9, n_iter=2500,
                             experiment_name="(G)S6D9I2500")
