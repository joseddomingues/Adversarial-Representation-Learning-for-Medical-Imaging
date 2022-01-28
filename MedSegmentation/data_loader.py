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
        self.dirlist = [elem for elem in os.listdir(self.input_dir) if '_mask' not in elem]
        self.dirlist.sort()

    def __len__(self):
        return len(self.dirlist)

    def __getitem__(self, idx):
        img_id = self.dirlist[idx]

        temp = img_id.split(".")
        aux_mask_path = '.'.join(temp[:-1]) + "_mask." + temp[-1]

        i_path = os.path.join(self.input_dir, img_id)
        m_path = os.path.join(self.input_dir, aux_mask_path)

        image = cv2.imread(i_path)
        mask = cv2.imread(m_path, 0)
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            raise IOError('Image and Masks shape dont match')

        sample = {'image': image, 'masks': mask}

        if self.transform:
            sample = unet_augment(sample, vertical_prob=0.5, horizontal_prob=0.5)

        # Transpose required because conv reads (N,C,W,K)
        # /255 for normalization
        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['image'].astype(float)
        sample['image'] = sample['image'] / 255

        sample['masks'] = sample['masks'].reshape((mask.shape[0], mask.shape[1], 1)).transpose((2, 0, 1))
        sample['masks'].astype(float)
        sample['masks'] = sample['masks'] / 255

        return sample
