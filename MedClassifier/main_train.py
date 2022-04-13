from argparse import ArgumentParser
from datetime import datetime

import torch.cuda
import torch.nn as nn
import torchvision.transforms as tvt
from mlflow import log_param, log_metric, start_run
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from breast_dataset import BreastDataset
from mammogram_classifier import MammogramClassifier


def train_classifier(options_map, curr_device):
    now = datetime.now()

    with start_run(nested=True, run_name=now.strftime("%d_%m_%y_%H_%M_%S")):

        # Log parameters to mlflow
        print("Logging Parameters...", end=" ")
        log_param("N Iterations", options_map.iter)
        log_param("Image Size Training", "614x499")
        log_param("Batch Size", 3)
        log_param("Loss", "Cross Entropy Loss")
        log_param("Optimizer", "Adam")
        print("Done!")

        # Dimensions of the 25% size image
        reduced_images_size = (614, 499)

        # Initialize the dataset with the processing for the ResNet approach
        print("Dataset Preparations...", end=" ")
        transformations = tvt.Compose([
            tvt.Resize(reduced_images_size),
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = BreastDataset(data_root_folder=options_map.train_folder, transform=transformations)
        train_data = DataLoader(train_dataset, batch_size=30, shuffle=True, pin_memory=True)
        print("Done!")

        # Initialize the network
        # 3 classes -> Benign, Malign, Normal
        print("Creating Classifier...", end=" ")
        nnet = MammogramClassifier(n_classes=3)
        nnet.to(curr_device)
        print("Done!")

        # Create optimizer and loss function
        optimizer = torch.optim.Adam(nnet.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # Initiate tensorboard writer and start training
        writer = SummaryWriter("tensorboard_logs")
        nnet.train()

        print("Initiating Train")
        _iter = tqdm(range(options_map.iter))
        for epoch in _iter:
            _iter.set_description('Iter [{}/{}]:'.format(epoch + 1, options_map.iter))

            curr_loss = 0

            for i, batch in enumerate(train_data, 0):
                # Move batch to gpu
                images, labels = batch[0].to(curr_device), batch[1].to(curr_device)

                # Zero the gradients
                optimizer.zero_grad()

                # Classify batch, calculate loss and update
                pred = nnet(images)
                loss = loss_fn(pred, labels)
                curr_loss += loss.item()
                loss.backward()
                optimizer.step()

            # Write data to tensorboard
            writer.add_scalar("Loss/train", loss.item(), epoch + 1)
            log_metric('Train Loss', loss.item(), step=epoch + 1)
            current_grid = make_grid(images)
            writer.add_image("images", current_grid, epoch + 1)
            writer.add_graph(nnet, images)
            _iter.set_description(f"Avg Bacth Loss: {curr_loss / i + 1}")

        # Close the writer and save the model
        print("Train Finished. File saved to ./current_classifier.pth")
        model_path = "./current_classifier.pth"
        torch.save(nnet.state_dict(), model_path)
        writer.close()


if __name__ == "__main__":
    # Construct argument parser
    arg = ArgumentParser()
    arg.add_argument('--train_folder', help='Train Folder for Classification', type=str, required=True)
    arg.add_argument('--iter', help='Number of Iterations to Train', type=int, required=True)
    opt_map = arg.parse_args()

    # Get current device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_classifier(opt_map, device)
