import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)

class TT(nn.Module):
    def __init__(self):
        super(TT, self).__init__()
        self.linear=Linear(196608,1000)
    def forward(self,input):
        output=self.linear(input)
        return output

tt=TT()
step=0
writer=SummaryWriter("D:\Happy\Project\learn_pytorch\hymenoptera_data\logs\linear")
for data in dataloader:
    imgs,targets=data
    #torch.Size([64, 3, 32, 32])->[1,1,1,-1] 展开
    # output=torch.reshape(imgs,[1,1,1,-1])
    output=torch.flatten(imgs)#摊平 比output=torch.reshape(imgs,[1,1,1,-1])更简单 output.shape=[196608]
    # print(output.shape)  # 获得196608
    output_linear=tt(output)
    output_linear=torch.reshape(output_linear,[1,1,1,-1])
    writer.add_images("input",imgs,step)
    writer.add_images("output_linear",output_linear,step)
    step+=1
writer.close()

