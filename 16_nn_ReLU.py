
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64)
##数字演示
input=torch.tensor([[1,-0.5],
                    [-1,3]])
input=torch.reshape(input,[1,1,2,2])
print(input.shape)

class TT(nn.Module):
    def __init__(self):
        super(TT, self).__init__()
        self.relu=ReLU()
        self.sigmoid=Sigmoid()
    def forward(self,input):
        output1=self.relu(input)
        output2=self.sigmoid(input)
        return output2
tt=TT()
output=tt(input)
print(output)

writer=SummaryWriter(r"D:\Happy\Project\learn_pytorch\hymenoptera_data\logs\nonlinear")
step=0
for data in dataloader:
    imgs,targets=data
    imgs_output=tt(imgs)
    writer.add_images("input_1",imgs,step)
    writer.add_images("output_sigmoid",imgs_output,step)
    step+=1
writer.close()


