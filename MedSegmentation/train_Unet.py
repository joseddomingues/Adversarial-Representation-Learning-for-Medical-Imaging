import glob
import os
import re
import time
from argparse import ArgumentParser

import cv2
import pytorch_lightning.lite as LightningLite
import torch
import torch.nn as nn
from mlflow import log_param, log_metric, start_run
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset

from data_augment import unet_augment
from evaluate_segmentation import jaccard_index, dice_coeff
from networks import unet

# Create Summary Writter
writer = SummaryWriter()

# Create ArgParser
arg = ArgumentParser()

arg.add_argument('--n_iters', help='Number of base iterations', type=int, default=5000)
arg.add_argument('--batch_size', help='Batch Size', type=int, default=3)
arg.add_argument('--l_rate', help='Learning Rate', type=float, default=0.01)
arg.add_argument('--train_folder', help='Train Folder for Segmentation', type=str, default='train')
arg.add_argument('--val_folder', help='Validation Folder for Segmentation', type=str, default='validation')
arg.add_argument('--model_checkpoints', help='Folder For Model Checkpoints', type=str, default='model_checkpoints')
arg.add_argument('--optimizer_checkpoints', help='Folder For Model Optimizers', type=str, default='model_optimizers')
arg.add_argument('--graphs_dir', help='Folder For Graphs Model', type=str, default='graphs_unet')
arg.add_argument('--experiment_name', help='Experiment Name For MLFlow', type=str, default='Experiment_1')

opt_map = arg.parse_args()

# Assing parameters
batch_size = opt_map.batch_size  # mini-batch size
n_iters = opt_map.n_iters  # total iterations
learning_rate = opt_map.l_rate
train_directory = opt_map.train_folder
validation_directory = opt_map.val_folder
checkpoints_directory_unet = opt_map.model_checkpoints
optimizer_checkpoints_directory_unet = opt_map.optimizer_checkpoints
graphs_unet_directory = opt_map.graphs_dir
validation_batch_size = 1


class ImageDataset(Dataset):
    """
    Defining the class to load datasets
    """

    def __init__(self, input_dir='train', transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.dirlist = os.listdir(self.input_dir)
        self.dirlist.sort()

    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        img_id = self.dirlist[idx]

        image = cv2.imread(os.path.join(self.input_dir, img_id, "images", img_id + ".png"))

        input_net_1 = image
        input_net_2 = image
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = image.reshape((256, 256, 3))
        input_net_2 = cv2.resize(input_net_2, (128, 128), interpolation=cv2.INTER_CUBIC)

        mask_path = glob.glob(os.path.join(self.input_dir, img_id) + "/*.png")
        no_of_masks = int(mask_path[0].split("_")[1])

        masks = cv2.imread(mask_path[0], 0)
        masks = cv2.resize(masks, (256, 256), interpolation=cv2.INTER_CUBIC)
        masks = masks.reshape((256, 256, 1))

        sample = {'image': image, 'masks': masks}

        if self.transform:
            sample = unet_augment(sample, vertical_prob=0.5, horizontal_prob=0.5)

        # As transforms do not involve random crop, number of masks must stay the same
        sample['count'] = no_of_masks
        sample['image'] = sample['image'].transpose((2, 0,
                                                     1))  # The convolution function in pytorch expects data in format (N,C,H,W) N is batch size , C are channels H is height and W is width. here we convert image from (H,W,C) to (C,H,W)
        sample['masks'] = sample['masks'].reshape((256, 256, 1)).transpose((2, 0, 1))

        sample['image'].astype(float)
        sample['image'] = sample['image'] / 255  # image being rescaled to contain values between 0 to 1 for BCE Loss
        sample['masks'].astype(float)
        sample['masks'] = sample['masks'] / 255

        return sample


class Lite(LightningLite):
    def run(self):

        train_dataset = ImageDataset(input_dir=train_directory, transform=True)  # Training Dataset
        validation_dataset = ImageDataset(input_dir=validation_directory, transform=True)  # Validation Dataset

        num_epochs = n_iters / (len(train_dataset) / batch_size)
        num_epochs = int(num_epochs)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        batch_size=validation_batch_size,
                                                        shuffle=False)

        train_loader = self.setup_dataloaders(train_loader)
        validation_loader = self.setup_dataloaders(validation_loader)

        model = unet()  # R2U_Net()
        iteri = 0
        iter_new = 0

        # checking if checkpoints exist to resume training and create it if not
        if os.path.exists(checkpoints_directory_unet) and len(os.listdir(checkpoints_directory_unet)):
            checkpoints = os.listdir(checkpoints_directory_unet)
            checkpoints.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
            model = torch.load(checkpoints_directory_unet + '/' + checkpoints[-1])  # changed to checkpoints
            iteri = int(re.findall(r'\d+', checkpoints[-1])[0])  # changed to checkpoints
            iter_new = iteri
            print("Resuming from iteration " + str(iteri))

        elif not os.path.exists(checkpoints_directory_unet):
            os.makedirs(checkpoints_directory_unet)

        if not os.path.exists(graphs_unet_directory):
            os.makedirs(graphs_unet_directory)

        # Loss Class #BCE Loss has been used here to determine if the pixel belogs to class or not.(This is the case of segmentation of a single class)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer class
        model, optimizer = self.setup(model, optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                    gamma=0.1)  # this will decrease the learning rate by factor of 0.1 every 10 epochs

        # https://discuss.pytorch.org/t/can-t-import-torch-optim-lr-scheduler/5138/6

        if os.path.exists(optimizer_checkpoints_directory_unet) and len(
                os.listdir(optimizer_checkpoints_directory_unet)):
            checkpoints = os.listdir(optimizer_checkpoints_directory_unet)
            checkpoints.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
            optimizer.load_state_dict(torch.load(optimizer_checkpoints_directory_unet + '/' + checkpoints[-1]))
            print("Resuming Optimizer from iteration " + str(iteri))
        elif not os.path.exists(optimizer_checkpoints_directory_unet):
            os.makedirs(optimizer_checkpoints_directory_unet)

        beg = time.time()  # time at the beginning of training
        print("Training Started!")

        with start_run(nested=True, run_name=opt_map.experiment_name):
            # Log parameters to mlflow
            log_param("N Iterations", n_iters)
            log_param("Learning Rate", learning_rate)
            log_param("Training Batch Size", batch_size)
            log_param("Validation Batch Size", validation_batch_size)
            log_param("Num Epochs", num_epochs)

            for epoch in range(num_epochs):
                print("\nEPOCH " + str(epoch + 1) + " of " + str(num_epochs) + "\n")
                for i, datapoint in enumerate(train_loader):
                    datapoint['image'] = datapoint['image'].type(
                        torch.FloatTensor)  # typecasting to FloatTensor as it is compatible with CUDA
                    datapoint['masks'] = datapoint['masks'].type(torch.FloatTensor)

                    image = Variable(datapoint['image'])
                    masks = Variable(datapoint['masks'])

                    optimizer.zero_grad()  # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
                    outputs = model(image)
                    loss = criterion(outputs, masks)

                    # Log train metrics
                    log_metric("Dice Coeff Train",
                               dice_coeff(y_true=masks.detach().cpu(), y_pred=outputs.detach().cpu()),
                               step=epoch + 1)
                    log_metric("Jaccard Index Train",
                               jaccard_index(y_true=masks.detach().cpu(), y_pred=outputs.detach().cpu()),
                               step=epoch + 1)

                    self.backward(loss)  # Backprop
                    optimizer.step()  # Weight update
                    writer.add_scalar('Training Loss', loss.item(), iteri)
                    log_metric("Training Loss", loss.item(), step=epoch + 1)
                    iteri = iteri + 1
                    if iteri % 10 == 0 or iteri == 1:
                        # Calculate Accuracy
                        validation_loss = 0
                        total = 0
                        # Iterate through validation dataset
                        for j, datapoint_1 in enumerate(validation_loader):  # for validation
                            datapoint_1['image'] = datapoint_1['image'].type(torch.FloatTensor)
                            datapoint_1['masks'] = datapoint_1['masks'].type(torch.FloatTensor)

                            input_image_1 = Variable(datapoint_1['image'])
                            output_image_1 = Variable(datapoint_1['masks'])

                            # Forward pass only to get logits/output
                            outputs_1 = model(input_image_1)
                            validation_loss += criterion(outputs_1, output_image_1).item()

                            # Log validation metrics
                            log_metric("Dice Coeff Validation",
                                       dice_coeff(y_true=output_image_1.detach().cpu(),
                                                  y_pred=outputs_1.detach().cpu()),
                                       step=epoch + 1)
                            log_metric("Jaccard Index Validation",
                                       jaccard_index(y_true=output_image_1.detach().cpu(),
                                                     y_pred=outputs_1.detach().cpu()),
                                       step=epoch + 1)

                            total += datapoint_1['masks'].size(0)
                        validation_loss = validation_loss
                        log_metric("Validation Loss", validation_loss, step=epoch + 1)
                        writer.add_scalar('Validation Loss', validation_loss, iteri)
                        # Print Loss
                        time_since_beg = (time.time() - beg) / 60
                        print('Iteration: {}. Loss: {}. Validation Loss: {}. Time(mins) {}'.format(iteri, loss.item(),
                                                                                                   validation_loss,
                                                                                                   time_since_beg))
                scheduler.step()

            torch.save(model, checkpoints_directory_unet + '/model_iter_' + str(iteri) + '.pt')
            torch.save(optimizer.state_dict(),
                       optimizer_checkpoints_directory_unet + '/model_iter_' + str(iteri) + '.pt')
            print("model and optimizer saved at iteration : " + str(iteri))
            writer.export_scalars_to_json(graphs_unet_directory + "/all_scalars_" + str(
                iter_new) + ".json")  # saving loss vs iteration data to be used by visualise.py
            writer.close()

        # Save model to mlflow
        # set_tracking_uri("http://localhost:8000")
        # log_model(model, "model", "UNet_Segmentation_Model_1")


Lite(devices="auto", accelerator="auto").run()
