import torchvision
import torch
import torchvision.transforms as transforms

def loadtraindata():
    path = r"data/train_flower"                                         # 路径
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(cifar_norm_mean, cifar_norm_std),])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader

def loadtestdata():
    path = r"data/test_flower"
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    testset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(cifar_norm_mean, cifar_norm_std), ])
                                                )
    print(len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=True, num_workers=2)
    return testloader