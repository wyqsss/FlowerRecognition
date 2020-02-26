# FlowerRecognition
鲜花分类小项目
配置要求：
python3环境
pytorch
torchvision

数据来源： http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html

运行平台：GTX950

文件说明：
data文件夹用于存数据
models文件夹用于存神经网络模型
checkpoint文件夹用于存训练结果，由于pth文件比较大，只保留了一个典型的训练结果

代码及文本文件说明：
splitflower.py使用分配训练集和测试集的，初始数据集所有数据都在一个文件夹内
dataSet.py是使用senet时载入数据的文件
MyDataSet.py是使用全连接神经网络时载入数据的文件
pcaDataSet.py是原来用于PCA降维时的载入数据方式，现在是没用的
train.py是主体训练文件，里面也包含了测试模块，所以就没有单独在写测试文件
accuracy.txt存储了测试的准确率
loss.txt存储了训练的损失函数
time.txt存储了训练的时间

代码运行方式：
python train.py -net [网络模型] -ce [True|False] -weights [pth文件路径]
ce默认为false如何为true则表示继续上次训练，此时需要weights这个参数，否则不需要。
