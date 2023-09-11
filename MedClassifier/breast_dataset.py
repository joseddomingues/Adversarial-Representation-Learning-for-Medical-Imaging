import os

import torchvision.transforms as tvt
from PIL import Image
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    def __init__(self, data_root_folder, transform=None, augment=None):

        # Labels map
        self.labels_map = {"benign": 0, "malign": 1, "normal": 2}

        # For each folder on root get images
        self.images = []
        self.images_target = []

        for folder in os.listdir(data_root_folder):
            for curr_image in os.listdir(os.path.join(data_root_folder, folder)):

                if "_mask" in curr_image or curr_image.startswith("."):
                    continue

                curr_image_path = os.path.join(data_root_folder, folder, curr_image)

                # Reads the current image and preprocess it just in case
                target_image = Image.open(curr_image_path)

                if augment:
                    target_image = augment(target_image)
                else:
                    converter = tvt.ToTensor()
                    target_image = converter(target_image)

                if transform:
                    target_image = transform(target_image)

                self.images.append(target_image)
                self.images_target.append(self.labels_map[folder])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images_target[idx]
