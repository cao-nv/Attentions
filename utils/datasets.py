import numpy as np
import os
import cPickle
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import ipdb
RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)

Augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD)
])
Normalization = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD)])


def augment_data(images, augment=True):
    if augment:
        augmented_images = Augmentation(images)
        return augmented_images
    else:
        normalized_images = Normalization(images)
        return normalized_images


def unpicke(file):
    with open(file, 'rb') as fp:
        data_dict = cPickle.load(fp)
    return data_dict


class CIFAR100(Dataset):
    train_files = ['train']
    test_files = ['test']

    @staticmethod
    def load_data(data_dir, is_test=False):
        images = []
        labels = []
        files = CIFAR100.test_files if is_test else CIFAR100.train_files
        file_paths = [os.path.join(data_dir, batch_name) for batch_name in files]
        for path in file_paths:
            data_dict = unpicke(path)
            batch_images = data_dict['data']
            batch_labels = data_dict['fine_labels']
            images.append(batch_images.reshape(-1, 3, 32, 32))
            labels.append(batch_labels)
        images = np.concatenate(images)  # Channel first for Pytorch
        images = images.transpose((0, 2, 3, 1))  # For tensorflow default order
        labels = np.concatenate(labels)
        return images, labels

    @staticmethod
    def load_meta(data_dir):
        meta = unpicke(os.path.join(data_dir, 'meta'))
        return meta['coarse_label_names'], meta['fine_label_names']
        
    def __init__(self, data_dir, is_test=False, augmentation=True):
        super(CIFAR100, self).__init__()
        
        self.data_dir = data_dir
        self._is_test = is_test
        self.augmentation = augmentation
        
        self.images, self.labels = CIFAR100.load_data(self.data_dir, self._is_test)
        self.coarse_label_names, self.fine_label_names = CIFAR100.load_meta(self.data_dir)
        
    def __getitem__(self, index):
        image = self.images[index]
        image = augment_data(image, self.augmentation)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)

    
