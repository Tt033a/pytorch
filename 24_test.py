#test of a picture from Internet
import torch
import torchvision.transforms
from PIL import Image
from model import *

image_path="test_images/OIP-C.jpg"
image=Image.open(image_path)
print(image)#转变成PIL类型
image=image.convert("RGB")#转变成三个颜色通道符合输入要求

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])

image=transform(image)
image=torch.reshape(image,[1,3,32,32])#别忘了batchsize！！
print(image.shape)
#加载模型
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

    def forward(self,input):
        output=self.model(input)
        return output
#通过字典加载参数
tt=TT()
tt.load_state_dict(torch.load("tt_29.pth",map_location=torch.device('cpu')))

tt.eval()
with torch.no_grad():
    output=tt(image)
print(output)

print(output.argmax(1))
