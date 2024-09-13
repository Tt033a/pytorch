
from PIL import Image
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10("D:\Happy\Project\learn_pytorch\hymenoptera_data\CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#成组
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

writer=SummaryWriter("dataloader")
step=0
for epoch in range(2):
    for data in test_loader:
        imgs_t,targets=data#包括若干张图
        # print(targets)
        # print(imgs_t.shape)#获得张数，通道，大小
        writer.add_images("Epoch:{}".format(epoch),imgs_t,step,dataformats="NCHW")
        step=step+1

writer.close()
