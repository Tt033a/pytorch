#对CIFAR10进行分类
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=1,drop_last=True)

class TT(nn.Module):
    def __init__(self):
        super(TT, self).__init__()
        # self.conv2d1=Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2,stride=1)#要求图像大小不改变，需要套公式计算padding的值
        # self.conv2d2=Conv2d(32,32,5,padding=2)
        # self.conv2d3=Conv2d(32,64,5,padding=2)
        # self.maxpolling = MaxPool2d(kernel_size=2)
        # self.flatten=Flatten()
        # self.linear1=Linear(in_features=1024,out_features=64)
        # self.linear2=Linear(64,10)#分成10个类型
        #利用sequential简化网络
        self.model1=Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(64, 10)
        )

    def forward(self,input):#前向传播
        # x=self.conv2d1(input)
        # x=self.maxpolling(x)
        # x=self.conv2d2(x)
        # x=self.maxpolling(x)
        # x=self.conv2d3(x)
        # x=self.maxpolling(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)

        x=self.model1(input)
        return x
writer=SummaryWriter("logs\sequential")

tt=TT()
print(tt)#获得网络结构
loss=CrossEntropyLoss()

#自定义简单的数据进行检查并进行可视化
input=torch.ones((64,3,32,32))
output=tt(input)
print(output.shape)
writer.add_graph(tt,input)#tensorboard显示图表

#计算每张图的可能和loss
for data in dataloader:
    imgs,targets=data
    outputs=tt(imgs)
    # print(targets)
    # print(outputs)
    result_loss=loss(outputs,targets)
    print(result_loss)

    #反向传播降低Loss，注意变量
    result_loss.backward()#可以获得grad
    print("ok")

writer.close()