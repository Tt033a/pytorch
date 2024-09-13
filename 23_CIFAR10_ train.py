import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
import time#用于计时

#准备数据集
train_data=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("hymenoptera_data/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#获得数据集长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集长度:{}".format(train_data_size),
      "测试数据集长度:",test_data_size)

#加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

#创建网络模型
tt=TT()
if torch.cuda.is_available():
    tt=tt.cuda()#网络模型调用GPU

#创建损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

#创建优化器
#1e-2=0.01
learning_rate=0.01
optimizier=torch.optim.SGD(tt.parameters(),lr=learning_rate)

#设置训练网络的参数
#记录训练次数
total_train_step=0
#记录测试次数
total_test_step=0
#训练轮数
epoch=10
#可视化
writer=SummaryWriter("logs\CIFAR10")
start_time=time.time()#记录现在的时间
for i in range(epoch):
    print("---------第{}轮训练开始-------".format(i+1))
    #开始训练步骤
    for data in train_dataloader:
        imgs, targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        outputs=tt(imgs)
        loss=loss_fn(outputs,targets)

        #优化器调优
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()

        total_train_step+=1
        if total_train_step%100==0:
            end_time=time.time()
            print(end_time-start_time)#100batchsizes计算训练时间
            print("训练次数：{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #开始测试
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs=imgs.cuda()
                targets=targets.cuda()
            outputs=tt(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss
            #分类问题计算准确率
            preds=outputs.argmax(1)  # 获得最大得分的位置，如果print(outputs.argmax(0))则是矩阵纵向（列）来看哪个大
            accuracy=(preds==targets).sum()
            total_accuracy+=accuracy




    total_test_step+=1
    print("整体测试集的Loss：{}".format(total_test_loss))#这个参数取决于研究什么问题，在分类的问题上一般用
    print("整体测试集上的正确率：{}".format(total_accuracy.float() / test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy", (total_accuracy.float() / test_data_size), total_test_step)



    #保存每一次的模型参数
    torch.save(tt.state_dict(),"tt_{}.pth".format(total_test_step))
    print("模型已保存")
writer.close()

