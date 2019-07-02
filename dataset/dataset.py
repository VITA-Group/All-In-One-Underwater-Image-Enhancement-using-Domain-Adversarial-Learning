import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image

class NYUUWDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'test':
            self.uw_images = self.uw_images[self.test_start:self.test_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.cl_images = []

        for img in self.uw_images:
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img).split('_')[0]  + '.' + img_format))

        for uw_img, cl_img in zip(self.uw_images, self.cl_images):
            assert os.path.basename(uw_img).split('_')[0] == os.path.basename(cl_img).split('.')[0], ("Files not in sync.")

        self.transform = transforms.Compose([
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            cl_img = self.transform(Image.open(self.cl_images[index]))
            water_type = int(os.path.basename(self.uw_images[index]).split('_')[1])
            name = os.path.basename(self.uw_images[index])[:-4]

            return uw_img, cl_img, water_type, name             

    def __len__(self):
        return self.size

class UIEBDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'test':
            self.uw_images = self.uw_images[self.test_start:self.test_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        uw_img = self.transform(Image.open(self.uw_images[index]))

        return uw_img, -1, -1, os.path.basename(self.uw_images[index])

    def __len__(self):
        return self.size