''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        with h5py.File('./data/data.h5', 'r', driver='core') as data:
            if self.split == 'Training':
                self.train_data = np.array(
                    data['Training_pixel']).reshape((28709, 48, 48))
                self.train_labels = np.array(data['Training_label'])
            elif self.split == 'PublicTest':
                self.PublicTest_data = np.array(
                    data['PublicTest_pixel']).reshape((3589, 48, 48))
                self.PublicTest_labels = np.array(data['PublicTest_label'])

            else:
                self.PrivateTest_data = np.array(
                    data['PrivateTest_pixel']).reshape((3589, 48, 48))
                self.PrivateTest_labels = np.array(data['PrivateTest_label'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
