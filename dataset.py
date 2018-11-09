import random
from mxnet import image, gluon
from mxnet.gluon import nn
import os

class RandomCrop(nn.Block):
    def __init__(self, size):
        super(RandomCrop,self).__init__()
        self._size = size

    def forward(self,x):
        h, w, _ = x.shape
        th, tw = (self._size,self._size)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        out = x[i:i + th, j:j + tw, :]
        return out

class DataSet(gluon.data.Dataset):
    def __init__(self,root,crop_transform,downsample_transform,last_transform):
        self.dir = root
        self.paths = [os.path.join(self.dir,f) for f in os.listdir(self.dir)]
        self.paths = sorted(self.paths)
        self.crop_transform = crop_transform
        self.downsample_transform = downsample_transform
        self.last_transform = last_transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = image.imread(path)

        hr_img = self.crop_transform(img)
        lr_img = self.downsample_transform(hr_img)
        hr_img = self.last_transform(hr_img)
        lr_img = self.last_transform(lr_img)
        return hr_img,lr_img

    def __len__(self):
        return len(self.paths)

