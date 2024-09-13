import torchvision
from torch.utils.tensorboard import SummaryWriter
#定义批量处理方法
trans_con=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="D:\Happy\Project\learn_pytorch\hymenoptera_data\CIFAR10",train=True,transform=trans_con,download=True)#训练数据集，download=True 开启下载，批量处理图片
test_set=torchvision.datasets.CIFAR10(root="D:\Happy\Project\learn_pytorch\hymenoptera_data\CIFAR10",train=False,transform=trans_con,download=True)#测试数据集

# print(train_set[0])#得到图片和类型
# img,target=train_set[0]
# print(train_set.classes[target])
# img.show()
# print(train_set[0])
la=[]
writer=SummaryWriter("CIFAR10\p10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("p10",img,2*i)
    la.append(train_set.classes[target])
    print(train_set.classes[target])
writer.close()
print(la)