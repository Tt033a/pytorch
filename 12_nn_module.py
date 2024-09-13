import torch
import torch.nn as nn
import torch.nn.functional as F
class TT(nn.Module):#继承
    def __init__(self):
        super().__init__()#初始化

    def forward(self,input):
        output=input+1
        return output

tt=TT()
x=torch.tensor(1.0)
output=tt(x)
print(output)