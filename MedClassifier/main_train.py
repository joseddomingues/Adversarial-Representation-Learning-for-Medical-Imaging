from argparse import ArgumentParser

import torch.cuda
import torch.nn as nn
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np

from breast_dataset import BreastDataset
from mammogram_classifier import MammogramClassifier


class EarlyStopper:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='current_classifier.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_classifier(options_map, curr_device):
    # Dimensions of the 25% size image
    reduced_images_size = (614, 499)

    # Initialize the dataset with the processing for the ResNet approach
    print("Dataset Preparations...", end=" ")
    transformations = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tvt.Resize(reduced_images_size)
    ])

    # Data augmentation techniques
    augmentations = tvt.Compose([
        tvt.RandomHorizontalFlip(),
        tvt.RandomVerticalFlip(),
        tvt.RandomAdjustSharpness(2),
        tvt.RandomApply(transforms=[tvt.ColorJitter(brightness=0.5, hue=0.3)]),
        tvt.RandomApply(transforms=[tvt.GaussianBlur(kernel_size=(5, 5))]),
        tvt.RandomPerspective(),
        tvt.RandomAutocontrast()
    ])

    train_dataset = BreastDataset(data_root_folder=options_map.train_folder, transform=transformations,
                                  augment=augmentations)
    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    print("Done!")

    # Initialize the network
    # 3 classes -> Benign, Malign, Normal
    print("Creating Classifier...", end=" ")
    nnet = MammogramClassifier(n_classes=3)
    nnet.to(curr_device)
    print("Done!")

    # Create optimizer and loss function
    if options_map.optim == 'adam':
        optimizer = torch.optim.Adam(nnet.parameters())
    else:
        optimizer = torch.optim.SGD(nnet.parameters(), lr=1e-3, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    # Initiate tensorboard writer, early stopper and start training
    writer = SummaryWriter("tensorboard_train_logs")
    patience = 10
    early_stopper = EarlyStopper(patience=patience)
    nnet.train()

    print("Initiating Train")
    _iter = tqdm(range(options_map.iter))
    for epoch in _iter:
        _iter.set_description('Iter [{}/{}]:'.format(epoch + 1, options_map.iter))

        running_loss = 0.0
        running_corrects = 0
        total = 0

        for i, batch in enumerate(train_data, 0):
            # Move batch to gpu
            images, labels = batch[0].to(curr_device), batch[1].to(curr_device)

            # Zero the gradients
            optimizer.zero_grad()

            # Classify batch, calculate loss and update
            pred = nnet(images)
            loss = loss_fn(pred, labels)
            _, preds = torch.max(pred, 1)
            loss.backward()
            optimizer.step()

            # statistics
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        lr_scheduler.step()

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / total
        acc = 100 * running_corrects / total
        print(f'\nEpoch {epoch + 1} =====> Loss: {epoch_loss:.4f}   Accuracy: {acc} %')

        # Write data to tensorboard
        writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", acc, epoch + 1)
        current_grid = make_grid(images)
        writer.add_image("images", current_grid, epoch + 1)
        writer.add_graph(nnet, images)

        # Add loss to early stopper and check if stop
        early_stopper(epoch_loss, nnet)

        if early_stopper.early_stop:
            print("\n\nTRAIN STOPPED =====> CONVERGENCE ACHIEVED")
            print(f"DURING {patience} EPOCHS THE LOSS NEVER DECREASED")
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
    arg.add_argument('--optim', help='Optimizer to use', default='adam', choices=['adam', 'sgdm'])
    opt_map = arg.parse_args()

    # Get current device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_classifier(opt_map, device)
