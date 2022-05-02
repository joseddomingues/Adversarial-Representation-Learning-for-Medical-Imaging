import os

from torch.utils.data import Dataset
import torchvision.transforms as tvt
from PIL import Image


class BreastDataset(Dataset):
    def __init__(self, data_root_folder, transform=None, augment=None):

        # Labels map
        self.labels_map = {
            "benign": 0,
            "malign": 1,
            "normal": 2
        }

        # Grab global variables
        self.root_data = data_root_folder
        self.transform = transform
        self.augment = augment

        # For each folder on root get images
        self.images = []
        self.images_target = []

        for folder in os.listdir(self.root_data):
            for curr_image in os.listdir(os.path.join(self.root_data, folder)):
                if "_mask" not in curr_image:
                    curr_image_path = os.path.join(self.root_data, folder, curr_image)

                    # Reads the current image and preprocess it just in case
                    target_image = Image.open(curr_image_path)

                    if self.augment:
                        target_image = self.augment(target_image)
                    else:
                        converter = tvt.ToTensor()
                        target_image = converter(target_image)

                    if self.transform:
                        target_image = self.transform(target_image)

                    self.images.append(target_image)
                    self.images_target.append(self.labels_map[folder])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images_target[idx]