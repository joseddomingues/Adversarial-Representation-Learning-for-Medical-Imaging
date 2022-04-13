from argparse import ArgumentParser

import torch.cuda
import torch.nn as nn
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from breast_dataset import BreastDataset
from mammogram_classifier import MammogramClassifier


class EarlyStopper:
    def __init__(self, iter_threshold, min_change):
        """
        Class for the early stop of the training process
        @param iter_threshold: Number of iterations without variation
        @param min_change: Minimum variation to stop
        """
        self.losses = []
        self.last_variations = []
        self.max_iter = iter_threshold
        self.min_change = min_change

    def add_loss(self, loss):
        """
        Adds the loss for the current epoch
        @param loss: Loss to add
        @return: True if added correctly
        """
        if len(self.losses) == 0:
            self.losses.append(loss)
            self.last_variations.append(abs(loss))
            return True

        if len(self.losses) == self.max_iter:
            self.losses.pop(0)
            self.last_variations.pop(0)

        variation = abs(self.losses[-1] - loss)
        self.losses.append(loss)
        self.last_variations.append(variation)
        return True

    def stop_train(self):
        """
        Gets if the train should stop or proceed
        @return: True if train should stop | False if not
        """
        if len(self.last_variations) < self.max_iter:
            return False

        if max(self.last_variations) <= self.min_change:
            return True


def train_classifier(options_map, curr_device):
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

    # Initiate tensorboard writer, early stopper and start training
    writer = SummaryWriter("tensorboard_train_logs")
    iter_var = 10
    var_change = 0.0001
    early_stopper = EarlyStopper(iter_threshold=iter_var, min_change=var_change)
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
        curr_loss = curr_loss / (i + 1)
        writer.add_scalar("Loss/train", curr_loss, epoch + 1)
        current_grid = make_grid(images)
        writer.add_image("images", current_grid, epoch + 1)
        writer.add_graph(nnet, images)
        print(f"Avg Bacth Loss: {curr_loss}")

        # Add loss to early stopper and check if stop
        early_stopper.add_loss(curr_loss)

        if early_stopper.stop_train():
            print("\n\nTRAIN STOPPED =====> CONVERGENCE ACHIEVED")
            print(f"DURING {iter_var} EPOCHS THE LOSS VARIED LESS THAN {var_change}")
            break

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
