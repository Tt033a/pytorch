import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer, SummaryWriter

dataset=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class TT(nn.Module):
    def __init__(self):
        super().__init__()#初始化
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)#定义卷积函数


    def forward(self,input):
        output=self.conv1(input)#输入input
        return output

tt=TT()#定义卷积函数
writer=SummaryWriter("D:\Happy\Project\learn_pytorch\hymenoptera_data\CIFAR10\conv2d")
step=0
for data in dataloader:
    imgs,targets=data

    # print(imgs.shape) #torch.Size([64, 3, 32, 32])

    imgs_tt=tt(imgs)
    # print(imgs_tt.shape) #torch.Size([64, 6, 30, 30])

    #torch.Size([64, 6, 30, 30])->[...,3,30,30]
    imgs_tt=torch.reshape(imgs_tt,[-1,3,30,30])#第一个数不知道是多少就写-1,变成可识别的图片
    print(imgs_tt.shape)

    writer.add_images("input",imgs,step)
    writer.add_images("output",imgs_tt, step)
    step += 1

writer.close()
