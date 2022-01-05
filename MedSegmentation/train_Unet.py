import os
import re
import time
from argparse import ArgumentParser

import torch
from mlflow import log_param, log_metric, start_run
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset

from losses import calc_loss
from data_loader import ImageDataset
from metrics import jaccard_index, dice_coeff
from networks import U_Net

# Create Summary Writter
writer = SummaryWriter()

#######################################################
# Create Arg parser
#######################################################
arg = ArgumentParser()

arg.add_argument('--n_epochs', help='Number of epochs', type=int, default=2000)
arg.add_argument('--batch_size', help='Batch Size', type=int, default=3)
arg.add_argument('--l_rate', help='Learning Rate', type=float, default=0.001)
arg.add_argument('--train_folder', help='Train Folder for Segmentation', type=str, default='train')
arg.add_argument('--val_folder', help='Validation Folder for Segmentation', type=str, default='validation')
arg.add_argument('--model_checkpoints', help='Folder For Model Checkpoints', type=str, default='model_checkpoints')
arg.add_argument('--optimizer_checkpoints', help='Folder For Model Optimizers', type=str, default='model_optimizers')
arg.add_argument('--experiment_name', help='Experiment Name For MLFlow', type=str, default='Experiment_1')

opt_map = arg.parse_args()

#######################################################
# Assigning Parameters
#######################################################
batch_size = opt_map.batch_size
num_epochs = opt_map.n_epochs
learning_rate = opt_map.l_rate
train_directory = opt_map.train_folder
validation_directory = opt_map.val_folder
checkpoints_directory_unet = opt_map.model_checkpoints
optimizer_checkpoints_directory_unet = opt_map.optimizer_checkpoints
validation_batch_size = 1

#######################################################
# Checking for GPU
#######################################################
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

pin_memory = False
if train_on_gpu:
    pin_memory = True

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
# Create Datasets
#######################################################
train_dataset = ImageDataset(input_dir=train_directory, transform=True)
validation_dataset = ImageDataset(input_dir=validation_directory, transform=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=pin_memory)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=validation_batch_size,
                                                shuffle=False,
                                                pin_memory=pin_memory)

#######################################################
# Create Model
#######################################################
model = U_Net()  # R2U_Net()
model.to(device)
new_e = 0

#######################################################
# Resume Model
#######################################################
if os.path.exists(checkpoints_directory_unet) and len(os.listdir(checkpoints_directory_unet)) >= 1:
    checkpoints = os.listdir(checkpoints_directory_unet)
    checkpoints.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
    model = torch.load(os.path.join(checkpoints_directory_unet, checkpoints[-1]))  # changed to checkpoints
    new_e = int(re.findall(r'\d+', checkpoints[-1])[0])  # changed to checkpoints
    num_epochs = num_epochs - new_e
    print("Resuming from iteration: " + str(new_e) + "\nMissing Iterations: " + str(num_epochs))

elif not os.path.exists(checkpoints_directory_unet):
    os.makedirs(checkpoints_directory_unet)

#######################################################
# Using Adam as Optimizer
#######################################################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # try SGD
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.99)

MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-5)
# scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)
# This will decrease the learning rate by factor of 0.1 every 10 epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#######################################################
# Resuming Optimizer
#######################################################

if os.path.exists(optimizer_checkpoints_directory_unet) and len(os.listdir(optimizer_checkpoints_directory_unet)) >= 1:
    checkpoints = os.listdir(optimizer_checkpoints_directory_unet)
    checkpoints.sort(key=lambda x: int((x.split('_')[2]).split('.')[0]))
    optimizer.load_state_dict(torch.load(os.path.join(optimizer_checkpoints_directory_unet, checkpoints[-1])))
    print("Resuming Optimizer from epoch " + str(new_e))

elif not os.path.exists(optimizer_checkpoints_directory_unet):
    os.makedirs(optimizer_checkpoints_directory_unet)

#######################################################
# Starts Training
#######################################################

beg = time.time()  # time at the beginning of training
print("Training Started!")

with start_run(nested=True, run_name=opt_map.experiment_name):
    # Log parameters to mlflow
    log_param("Learning Rate", learning_rate)
    log_param("Batch Size", batch_size)
    log_param("Num Epochs", num_epochs)

    for epoch in range(num_epochs):

        print("\nEPOCH " + str(epoch + 1) + "/" + str(num_epochs) + "\n")

        train_loss = 0.0
        valid_loss = 0.0

        model.train()

        for i, datapoint in enumerate(train_loader):
            # typecasting to FloatTensor as it is compatible with CUDA
            datapoint['image'] = datapoint['image'].type(torch.FloatTensor)
            datapoint['masks'] = datapoint['masks'].type(torch.FloatTensor)

            image = Variable(datapoint['image'].to(device))
            masks = Variable(datapoint['masks'].to(device))

            # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
            optimizer.zero_grad()
            outputs = model(image)
            loss = calc_loss(outputs, masks, 0.1)
            train_loss += loss.item()

            # Log train metrics
            log_metric("Dice Coeff Train", dice_coeff(y_true=masks.detach().cpu(), y_pred=outputs.detach().cpu()),
                       step=epoch + 1)
            log_metric("Jaccard Index Train", jaccard_index(y_true=masks.detach().cpu(), y_pred=outputs.detach().cpu()),
                       step=epoch + 1)

            # Backprop & Weight update
            loss.backward()
            optimizer.step()

        model.eval()
        torch.no_grad()

        # Iterate through validation dataset
        for j, datapoint_1 in enumerate(validation_loader):  # for validation
            datapoint_1['image'] = datapoint_1['image'].type(torch.FloatTensor)
            datapoint_1['masks'] = datapoint_1['masks'].type(torch.FloatTensor)

            input_image_1 = Variable(datapoint_1['image'].to(device))
            output_image_1 = Variable(datapoint_1['masks'].to(device))

            # Forward pass only to get logits/output
            outputs_1 = model(input_image_1)
            loss = calc_loss(outputs_1, output_image_1, 0.1)
            valid_loss += loss.item()

            # Log validation metrics
            log_metric("Dice Coeff Validation",
                       dice_coeff(y_true=output_image_1.detach().cpu(), y_pred=outputs_1.detach().cpu()),
                       step=epoch + 1)
            log_metric("Jaccard Index Validation",
                       jaccard_index(y_true=output_image_1.detach().cpu(), y_pred=outputs_1.detach().cpu()),
                       step=epoch + 1)

        scheduler.step()

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(validation_loader)

        # Write loss & log
        writer.add_scalar('Training Loss', train_loss, epoch + 1)
        log_metric("Training Loss", train_loss, step=epoch + 1)

        log_metric("Validation Loss", valid_loss, step=epoch + 1)
        writer.add_scalar('Validation Loss', valid_loss, epoch + 1)

        time_since_beg = (time.time() - beg) / 60
        print('Iteration: {}. Loss: {}. Validation Loss: {}. Time(mins) {}'.format(epoch, train_loss, valid_loss,
                                                                                   time_since_beg))

    torch.save(model, os.path.join(checkpoints_directory_unet, 'model_iter_' + str(epoch) + '.pt'))
    torch.save(optimizer.state_dict(),
               os.path.join(optimizer_checkpoints_directory_unet, 'model_iter_' + str(epoch) + '.pt'))
    print("Model and optimizer saved at epoch : " + str(epoch))

    writer.close()

    # Save model to mlflow
    # set_tracking_uri("http://localhost:8000")
    # log_model(model, "model", "UNet_Segmentation_Model_1")
