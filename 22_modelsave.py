import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

vgg16=torchvision.models.vgg16(weights=None)
#保存方式1 模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")
#保存方式2 模型参数（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#自己搭建的网络
class TT(nn.Module):
    def __init__(self):
        super().__init__()#初始化
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)#定义卷积函数


    def forward(self,input):
        output=self.conv1(input)#输入input
        return output
tt=TT()
torch.save(tt,"tt_method1.pth")
