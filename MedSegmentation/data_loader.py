import os

import cv2
from torch.utils.data import Dataset

from data_augment import unet_augment


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

        folder_contents = os.listdir(os.path.join(self.input_dir, img_id))
        i_path = [elem for elem in folder_contents if 'mask' not in elem][0]
        m_path = [elem for elem in folder_contents if 'mask' in elem][0]

        image = cv2.imread(i_path)
        mask = cv2.imread(m_path, 0)
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

        if mask.shape != image.shape:
            raise IOError('Image and Masks shape dont match')

        sample = {'image': image, 'masks': mask}

        if self.transform:
            sample = unet_augment(sample, vertical_prob=0.5, horizontal_prob=0.5)

        # Transpose required because conv reads (N,C,W,K)
        # /255 for normalization
        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['image'].astype(float)
        sample['image'] = sample['image'] / 255

        sample['masks'] = sample['masks'].transpose((2, 0, 1))
        sample['masks'].astype(float)
        sample['masks'] = sample['masks'] / 255

        return sample
