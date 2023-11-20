from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch
from torchvision import transforms
import torch.utils.data as td

CUT_SIZE = 44

def prepare_dataset(batch_size:int, fold: int):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(CUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    def transform_ten_crop(crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

    transform_test = transforms.Compose([
        transforms.TenCrop(CUT_SIZE),
        transforms.Lambda(transform_ten_crop),
    ])

    trainset = CK(split = 'Training', fold = fold, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testset = CK(split = 'Testing', fold = fold, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)

    return trainloader, testloader

class CK(td.Dataset):
    """`CK+ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,8,9,21,9,24,6 images for testing
        the split are in order according to the fold number
    """

    def __init__(self, split='Training', fold = 1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = fold # the k-fold cross validation
        with h5py.File('./data/CK_data.h5', 'r', driver='core') as data:
            number = len(data['data_label']) #981
            sum_number = [0,135,312,387,594,678,927,981] # the sum of class number
            test_number = [12,18,9,21,9,24,6] # the number of each class

            test_index = []
            train_index = []

            for j in range(len(test_number)):
                for k in range(test_number[j]):
                    if self.fold != 10: #the last fold start from the last element
                        test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
                    else:
                        test_index.append(sum_number[j+1]-1-k)

            for i in range(number):
                if i not in test_index:
                    train_index.append(i)

            print(len(train_index),len(test_index))

            # now load the picked numpy arrays
            if self.split == 'Training':
                self.train_data = []
                self.train_labels = []
                for ind in range(len(train_index)):
                    self.train_data.append(data['data_pixel'][train_index[ind]])
                    self.train_labels.append(data['data_label'][train_index[ind]])

            elif self.split == 'Testing':
                self.test_data = []
                self.test_labels = []
                for ind in range(len(test_index)):
                    self.test_data.append(data['data_pixel'][test_index[ind]])
                    self.test_labels.append(data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]
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
        elif self.split == 'Testing':
            return len(self.test_data)

