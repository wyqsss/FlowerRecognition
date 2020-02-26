import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import senet,linear1
import MyDataSet
import pcaDataSet
import dataSet
from time import localtime,strftime
CHECKPOINT_PATH = "checkpoint"
SAVE_EPOCH = 10
MILESTONES = [60, 120, 160]

def train(epoch):
    f = open("time.txt","a")
    f.write("开始时间："+strftime("%Y-%m-%d %H:%M:%S", localtime()))
    f.write("\n")
    f.close()
    net.train()
    ncount = 0
    sum = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        ncount += 1
        images = Variable(images)
        labels = Variable(labels)
        print(labels)
        print(images.shape)
        labels = labels.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        images = images
        outputs = net(images)
        print(outputs.shape)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_loader.dataset)
        ))

        sum = sum + loss.item()
    avg_loss = sum / ncount
    fw = open("loss.txt", "a")
    fw.write(str(avg_loss))
    fw.write("\n")
    fw.close()
    f = open("time.txt","a")
    f.write("结束时间："+strftime("%Y-%m-%d %H:%M:%S", localtime()))
    f.write("\n")
    f.close()



def eval_training(epoch):  # 用来计算平均损失和平均准确率
    net.eval()
    f1 = open("accuracy.txt", "a")
    test_loss = 0.0  # cost function error
    correct = 0.0
    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        print(labels)
        print(type(labels))
        images = images.cuda()
        labels = labels.cuda()
        images = images
        outputs = net(images)
        loss = loss_function(outputs, labels)  # 损失函数
        test_loss += loss.item()  # 张量里的元素值
        _, preds = outputs.max(1)  # 最大值的index
        correct += preds.eq(labels).sum()  # 统计相等的个数

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))

    f1.write('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()
    f1.close()

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-ce', type=bool, default=False, help='if continue train')
    parser.add_argument('-weights', type=str, default="", help='the weights file you want to test')

    args = parser.parse_args()

    print(args.net)

    if(args.ce == True and args.net == "senet18"):
        net = senet.seresnet18()
        net.load_state_dict(torch.load(args.weights))
        print("load")
        train_loader = MyDataSet.loadtraindata()
        test_loader = MyDataSet.loadtestdata()
    elif(args.ce == True and args.net == "linear"):
        net = linear1.getLinear1()
        net.load_state_dict(torch.load(args.weights))
        train_loader = dataSet.get_train_loader()
        test_loader = dataSet.get_test_loader()
    elif(args.net == "senet18"):
        net = senet.seresnet18()
        train_loader = MyDataSet.loadtraindata()
        test_loader = MyDataSet.loadtestdata()
    elif(args.net == "linear"):
        net = linear1.getLinear1()
        train_loader = dataSet.get_train_loader()
        test_loader = dataSet.get_test_loader()
    net = net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(train_loader)
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())).replace(':', '-')
    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.net, t)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, 200):
        train(epoch)  # 训练
        acc = eval_training(epoch)

        if  best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            print(checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            continue

        if not epoch % SAVE_EPOCH:
            print("保存")
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            print(checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

