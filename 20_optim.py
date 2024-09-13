import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=1)

class TT(nn.Module):
    def __init__(self):
        super(TT, self).__init__()
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
        x=self.model1(input)
        return x

tt=TT()
# print(tt)#获得网络结构TT(
#   # (model1): Sequential(
#   #   (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   #   (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   #   (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   #   (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   #   (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   #   (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   #   (6): Flatten(start_dim=1, end_dim=-1)
#   #   (7): Linear(in_features=1024, out_features=64, bias=True)
#   #   (8): Linear(in_features=64, out_features=10, bias=True)
#   # )
loss=CrossEntropyLoss()
#定义优化器
optim=torch.optim.SGD(tt.parameters(),lr=0.01)

#进行20轮学习
for epoch in range(20):
    running_loss=0.0
    #计算每张图的可能和loss——对数据进行一轮学习
    for data in dataloader:
        imgs,targets=data
        outputs=tt(imgs)
        result_loss=loss(outputs,targets)

        optim.zero_grad()  # 清零上一步计算梯度
        #反向传播降低Loss，注意变量
        result_loss.backward()#反向传播求得每个节点grad梯度
        optim.step()#参数调优
        running_loss=running_loss+result_loss
    print(running_loss)#逐渐变小
    # tensor(18852.9258, grad_fn= < AddBackward0 >)
    # tensor(16180.6963, grad_fn= < AddBackward0 >)