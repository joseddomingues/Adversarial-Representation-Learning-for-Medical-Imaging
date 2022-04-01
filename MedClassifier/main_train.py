from argparse import ArgumentParser

import torch.cuda
import torch.nn as nn
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from MedClassifier.breast_classifier import BreastClassifier
from MedClassifier.mammogram_classifier import MammogramClassifier
from MedClassifier.breast_dataset import BreastDataset


def train_classifier(options_map, curr_device):
    # Initialize the dataset with the processing for the ResNet approach
    transformations = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BreastDataset(data_root_folder=options_map["train_folder"], transform=transformations)
    train_data = DataLoader(train_dataset, batch_size=5, shuffle=True, pin_memory=True)

    # Initialize the network
    # 4 classes -> Benign, Malign, Normal
    nnet = MammogramClassifier(n_classes=3)
    nnet.to(curr_device)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(nnet.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initiate tensorboard writer and start training
    writer = SummaryWriter("tensorboard_logs")
    iter_log = 0
    nnet.train()

    for epoch in tqdm(range(options_map["iter"])):

        print(f"========== ITER {epoch + 1} ==========")

        for i, batch in enumerate(train_data, 0):
            # Move batch to gpu
            images, labels = batch[0].to(curr_device), batch[1].to(curr_device)

            # Zero the gradients
            optimizer.zero_grad()

            # Classify batch, calculate loss and update
            pred = nnet(images)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.update()

            # Write data to tensorboard
            writer.add_scalar("Loss/train", loss.item(), iter_log)
            current_grid = make_grid(images)
            writer.add_image("images", current_grid, iter_log)
            writer.add_graph(nnet, images)

            # Print data from current batch. Each 100 batches prints results
            if i % 10 == 0:
                print(f"Epoch: {epoch + 1} / Batch: {i + 1} => Loss: {loss.item()}")

            iter_log += 1

        iter_log += 1

    # Close the writer and save the model
    model_path = "./current_classifier.pth"
    torch.save(nnet.state_dict(), model_path)
    writer.close()

    # If a test set is given then evaluate the accuracy of the model
    if options_map["test_folder"]:
        test_dataset = BreastDataset(data_root_folder=options_map["test_folder"], transform=transformations)
        test_data = DataLoader(test_dataset, batch_size=5, shuffle=False, pin_memory=True)

        # Load the trained and saved model
        nnet = BreastClassifier()
        nnet.load_state_dict(torch.load(model_path))
        nnet.to(curr_device)

        # Evaluate each image batch
        classes = ["benign", "malign", "normal"]
        correct = 0
        total = 0
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        nnet.eval()
        with torch.no_grad():
            for batch in test_data:
                images, labels = batch[0].to(curr_device), batch[1].to(curr_device)
                pred = nnet(images)
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # collect the correct predictions for each class
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == "__main__":
    # Construct argument parser
    arg = ArgumentParser()
    arg.add_argument('--train_folder', help='Train Folder for Classification', type=str, required=True)
    arg.add_argument('--test_folder', help='Test Folder for Classification', type=str)
    arg.add_argument('--iter', help='Number of Iterations to Train', type=int)
    opt_map = arg.parse_args()

    # Get current device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_classifier(opt_map, device)
