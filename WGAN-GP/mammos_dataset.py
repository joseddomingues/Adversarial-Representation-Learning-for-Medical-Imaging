import os

import PIL.Image as Image
from torch.utils.data import Dataset


class MammographyDataset(Dataset):

    def __init__(self, base_folder, transformations):
        self.base_folder = base_folder
        self.transformations = transformations

        self.images = [os.path.join(self.base_folder, elem) for elem in os.listdir(self.base_folder) if
                       not elem.startswith(".")]

        self.processed_images = []
        if self.transformations:
            for elem in self.images:
                curr = Image.open(elem)
                self.processed_images.append(self.transformations(curr))

    def __getitem__(self, idx):
        # The label in this case is irrelevant since what we want is to generate images
        return self.images[idx], "Mammography"

    def __len__(self):
        return len(self.images)
