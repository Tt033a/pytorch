import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=121)
##数字演示
input=torch.tensor([[1.0,2,0,3,1],
       [0,1,2,3,1],
       [1,2,1,0,0],
       [5,2,3,1,1],
       [2,1,0,1,1]])
print(input.shape)
input=torch.reshape(input,[-1,5,5])
print(input.shape)

class TT(nn.Module):
    def __init__(self):
        super(TT, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

tt=TT()
output=tt(input)
print(output)

#图片演示
writer=SummaryWriter("D:\Happy\Project\learn_pytorch\hymenoptera_data\logs\maxpool")
step=0
for data in dataloader:
    imgs,targets=data
    imgs_output=tt(imgs)
    writer.add_images("input1",imgs,step)
    writer.add_images("output1",imgs_output, step)
    step += 1

writer.close()
