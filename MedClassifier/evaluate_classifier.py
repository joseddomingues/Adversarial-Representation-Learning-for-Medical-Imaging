from argparse import ArgumentParser

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as tvt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from breast_dataset import BreastDataset
from mammogram_classifier import MammogramClassifier


def process_pipeline_images(augment, transform, im_path):
    """

    @param augment:
    @param transform:
    @param im_path:
    @return:
    """
    target_image = Image.open(im_path)

    if augment:
        target_image = augment(target_image)
    else:
        converter = tvt.ToTensor()
        target_image = converter(target_image)

    if transform:
        target_image = transform(target_image)

    return target_image.numpy()


def evaluate_classifier(options_map, curr_device):
    # Dimensions of the 25% size image
    reduced_images_size = (614, 499)

    # Initialize the dataset with the processing for the ResNet approach
    print("Dataset Preparations...", end=" ")
    transformations = tvt.Compose([
        tvt.Resize(reduced_images_size),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data augmentation techniques
    # NOTE: ADDED BECAUSE OF THE LACK OF DATA FROM THE PIPELINE. THIS ENSURES IMAGES WILL BE DIFFERENT FROM EACH OTHER
    augmentations = tvt.Compose([
        tvt.ToTensor(),
        tvt.RandomHorizontalFlip(),
        tvt.RandomVerticalFlip(),
        tvt.RandomAdjustSharpness(2),
        tvt.RandomApply(transforms=[
            tvt.ColorJitter(brightness=[0.5, 0.99], hue=[0.3, 0.5], contrast=[0.5, 0.99], saturation=[0.5, 0.99])]),
        tvt.RandomApply(transforms=[tvt.GaussianBlur(kernel_size=(5, 5))]),
        tvt.RandomPerspective()
    ])

    test_dataset = BreastDataset(data_root_folder=options_map.test_folder)

    test_data = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)
    print("Done!")

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
    print("Done!")

    print("Initiate Testing")
    running_corrects = 0
    total = 0
    y_trues = []
    y_preds = []
    with torch.no_grad():

        for batch in test_data:
            # Process batch and move it to GPU
            images = []
            labels = batch[1].to(curr_device)

            for j in range(len(batch[0])):
                curr_image = process_pipeline_images(augmentations, transformations, batch[0][j])
                images.append(curr_image)
            images = np.array(images)
            images = torch.tensor(images, device=curr_device)

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
    print(f'Accuracy of the network on the 10000 test images: {100 * running_corrects / total} %')

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

    evaluate_classifier(opt_map, device)
