import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d

input=torch.tensor([[1.0,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
#构建卷积核
kernel=torch.tensor([[1.0,2,1],
                     [0,1,0],
                     [2,1,0]])
#变化input类型以满足卷积要求
input=torch.reshape(input,[1,1,5,5])
kernel=torch.reshape(kernel,[1,1,3,3])

print(input.shape)
print(kernel.shape)

output1=F.conv2d(input,kernel,stride=2)#二维数据处理
print(output1)

output2=F.conv2d(input,kernel,stride=2,padding=1)
print(output2)

class TT(nn.Module):
    def __init__(self):
        super().__init__()#初始化
        self.conv1=Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=1,padding=0)#定义卷积函数


    def forward(self,input):
        output=self.conv1(input)#输入input
        return output

tt=TT()#定义卷积函数
output3=tt(input)
print(output3)
#tensor([[[[-0.9357, -1.5658, -0.0192],
          # [-1.1589, -1.4552, -1.4039],
          # [-1.6457, -0.5955, -1.2162]]]], grad_fn=<ConvolutionBackward0>)
