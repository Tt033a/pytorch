import torch
from torch import nn

#搭建网络
class TT(nn.Module):
    def __init__(self):
        super(TT, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,input):#前向传播
        output=self.model(input)
        return output

#测试网络正确性——尺寸测试
if __name__ == '__main__':
    tt=TT()
    input1=torch.ones((64,3,32,32))#64张图片，每个位置都用1填充
    # input2=torch.tensor([[1,2],
    #                    [2,3]])
    output=tt(input1)
    # print(output.shape)#torch.Size([64, 10]) 表示返回64行数据，每行有10个数据，数据表示图片在10个类里面的概率
    # print(output)
