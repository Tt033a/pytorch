import torchvision

#加载参数
from torch import nn

vgg16_false=torchvision.models.vgg16(weights=None)
vgg16_true=torchvision.models.vgg16()
print(vgg16_true)#out_features=1000

#数据集只把数据分成10类【对应的数据集太大了】-改动网络，进行模型迁移
train_data=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#方法一，增加一个线性层，加到了VGG16里面
vgg16_true.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

#方法二：增加到(classifier)中
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

#修改模型
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)