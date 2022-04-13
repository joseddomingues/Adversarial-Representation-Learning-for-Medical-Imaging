from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from breast_dataset import BreastDataset
from mammogram_classifier import MammogramClassifier


def evaluate_classifier(options_map, curr_device):
    # Dimensions of the 25% size image
    reduced_images_size = (614, 499)

    # Initialize the dataset with the processing for the ResNet approach
    print("Dataset Preparations...", end=" ")
    transformations = tvt.Compose([
        tvt.Resize(reduced_images_size),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = BreastDataset(data_root_folder=options_map.test_folder, transform=transformations)
    test_data = DataLoader(test_dataset, batch_size=30, shuffle=False, pin_memory=True)
    print("Done!")

    # Evaluate each image batch
    classes = ["benign", "malign", "normal"]
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Initiates the loss functions
    loss_fn = nn.CrossEntropyLoss()

    # Load the classifier
    print("Loading Classifier...", end=" ")
    nnet = MammogramClassifier(n_classes=3)
    nnet.load_state_dict(torch.load(options_map.model_pth))
    nnet.to(curr_device)
    nnet.eval()
    print("Done!")

    writer = SummaryWriter("tensorboard_test_logs")
    iter_log = 0
    print("Initiate Testing")
    with torch.no_grad():
        for batch in test_data:

            images, labels = batch[0].to(curr_device), batch[1].to(curr_device)
            pred = nnet(images)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss = loss_fn(pred, labels)

            writer.add_scalar("Loss/test", test_loss.item(), iter_log)
            current_grid = make_grid(images)
            writer.add_image("images", current_grid, iter_log)
            writer.add_graph(nnet, images)
            iter_log += 1

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
    arg.add_argument('--test_folder', help='Test Folder for Classification', type=str, required=True)
    arg.add_argument('--model_pth', help='Model Path', type=str, required=True)
    opt_map = arg.parse_args()

    # Get current device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_classifier(opt_map, device)
