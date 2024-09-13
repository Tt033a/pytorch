

import torch
import torchvision.models
# import modelsave

#打开方式1对应保存方式1
from torch import nn
from torch.nn import Conv2d

model=torch.load("vgg16_method1.pth")

#打开方式2对应保存方式2【只有参数】
vgg16=torchvision.models.vgg16(weights=None)
#通过字典加载参数
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

#自己搭建的网络
class TT(nn.Module):
    def __init__(self):
        super().__init__()#初始化
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)#定义卷积函数


    def forward(self,input):
        output=self.conv1(input)#输入input
        return output
model=torch.load("tt_method1.pth")
print(model)