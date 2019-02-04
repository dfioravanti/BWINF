
import os
from PIL import Image

import numpy as np

import torch
import torch.utils.data as data


class MiniQuickDraw(data.Dataset):
    """Drop-in replacement for MNIST from the `QuickDraw <https://quickdraw.withgoogle.com/data/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``QuickDraw.npz`` exists
        train (bool, optional): If True, creates dataset from the training set
            otherwise from the test.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    meaning_labels = {
        0: 'soccer ball',
        1: 'map',
        2: 'rainbow',
        3: 'calendar',
        4: 'airplane',
        5: 'mountain',
        6: 'hospital',
        7: 'pizza',
        8: 'hand',
        9: 'cake'
    }

    filename = 'QuickDraw.npz'

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        data = np.load(os.path.join(self.root, self.filename))

        if self.train:
            self.xs = data['x_train']
            self.ys = data['y_train']
        else:
            self.xs = data['x_test']
            self.ys = data['y_test']

        self.xs = self.xs.reshape((-1, 28, 28))
        self.ys = self.ys.astype(np)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.xs[index], self.ys[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.xs)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))


if __name__ == '__main__':

    data = MiniQuickDraw('../data/')
    print(len(data))
    data = MiniQuickDraw('../data/', train=False)
    print(len(data))
