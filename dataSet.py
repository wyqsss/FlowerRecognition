import os
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image

import torch.utils.data as Data
import torch.utils.data.dataset as Dataset


train_path = "data/train_flower"
test_path = "data/test_flower"

cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)

transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])
rtransform = transforms.Compose([
                                                    transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])


class MyDataSet(Dataset.Dataset):#train数据
    def __init__(self,path):
        self.labels = []
        self.images = []
        ncount = 0
        for fname in os.listdir(path):
            lpath = os.path.join(path, fname)  # 每类图片的文件夹
            for iname in os.listdir(lpath):
                timg = Image.open(os.path.join(lpath, iname))
                timg = timg.convert("RGB")
                timg = transform(timg)
                timg = timg.view(-1)
                self.images.append(timg)
                label = ncount
                tlabel = torch.tensor(label, dtype=torch.long)
                print(tlabel)
                self.labels.append(tlabel)
            ncount += 1



    def __getitem__(self,idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


class MyDataSet_test(Dataset.Dataset):#test数据
    def __init__(self,path):
        self.labels = []
        self.images = []
        ncount = 0
        for fname in os.listdir(path):
            lpath = os.path.join(path, fname)  # 每类图片的文件夹
            img = []
            for iname in os.listdir(lpath):
                timg = Image.open(os.path.join(lpath, iname))
                timg = timg.convert("RGB")
                timg = rtransform(timg)
                timg = timg.view(-1)
                label = ncount
                tlabel = torch.tensor(label, dtype=torch.long)
                self.labels.append(tlabel)
                self.images.append(timg)
            ncount += 1

    def __getitem__(self,idx):
        return self.images[idx],self.labels[idx]

    def __len__(self):
        return len(self.images)


# filenames是训练数据文件名称列表，labels是标签列表


def get_train_loader():
    mydataset = MyDataSet(path=train_path)
    print(mydataset)
    dataloader = Data.DataLoader(dataset=mydataset,batch_size=4,num_workers=2,shuffle=True)
    return dataloader

def get_test_loader():#test数据集
    mydataset = MyDataSet_test(path=test_path)
    dataloader = Data.DataLoader(dataset=mydataset,batch_size=1,num_workers=1,shuffle=True)
    return dataloader

