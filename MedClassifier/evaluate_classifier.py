from argparse import ArgumentParser

import torch
import torchvision.transforms as tvt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from breast_dataset import BreastDataset
from mammogram_classifier import MammogramClassifier


def evaluate_classifier(options_map, curr_device):
    # Dimensions of the 25% size image
    reduced_images_size = (614, 499)

    # Initialize the dataset with the processing for the ResNet approach
    print("Dataset Preparations...", end=" ")
    transformations = tvt.Compose([
        tvt.Resize(reduced_images_size),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = BreastDataset(data_root_folder=options_map.test_folder, transform=transformations,
                                 augment=None)

    test_data = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)
    print("Dataset prepared")

    # Evaluate each image batch
    classes = ["benign", "malign", "normal"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Load the classifier
    print("Loading Classifier...", end=" ")
    nnet = MammogramClassifier(n_classes=3)
    nnet.load_state_dict(torch.load(options_map.model_pth))
    nnet.to(curr_device)
    nnet.eval()
    print("Model loaded")

    print("Initiate Testing")
    running_corrects = 0
    total = 0
    y_trues = []
    y_preds = []
    with torch.no_grad():

        for batch in test_data:
            images, labels = batch[0].to(curr_device), batch[1].to(curr_device)
            pred = nnet(images)
            _, predicted = torch.max(pred.data, 1)
            y_trues += list(labels.data)
            y_preds += list(predicted)
            running_corrects += torch.sum(predicted == labels.data)
            total += labels.size(0)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print(f'Accuracy of the network on the test images: {100 * running_corrects / total} %')

    y_trues = [elem.item() for elem in y_trues]
    y_preds = [elem.item() for elem in y_preds]

    print(f"Accuracy: {accuracy_score(y_trues, y_preds)}")
    print(f"Precision: {precision_score(y_trues, y_preds, average='weighted')}")
    print(f"Recall: {recall_score(y_trues, y_preds, average='weighted')}")
    print(f"F1: {f1_score(y_trues, y_preds, average='weighted')}")
    print(f"Matthews Corrcoef: {matthews_corrcoef(y_trues, y_preds)}")


if __name__ == "__main__":
    # Construct argument parser
    arg = ArgumentParser()
    arg.add_argument('--test_folder', help='Test Folder for Classification', type=str, required=True)
    arg.add_argument('--model_pth', help='Model Path', type=str, required=True)
    opt_map = arg.parse_args()

    # Get current device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate classifier
    evaluate_classifier(opt_map, device)
