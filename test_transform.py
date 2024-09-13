from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
img_path=r"D:\Happy\Project\learn_pytorch\hymenoptera_data\dataset\train\ants\49375974_e28ba6f17e.jpg"
img=Image.open(img_path)
print(img)
writer=SummaryWriter("logs")

#ToTensor 使用
tensor_trans=transforms.ToTensor()#实例化
img_tensor=tensor_trans(img)
writer.add_image("T",img_tensor,1)


#Normalize
print(img_tensor[0][0][0])#第一个通道，第一行，第一列
trans_norm=transforms.Normalize([0,0,0],[2,2,2])#规定标准化的均值和方差，因为有3个通道所以用了list
img_norm=trans_norm(img_tensor)
writer.add_image("T_Normalized",img_norm,3)
trans_norm1=transforms.Normalize([6,2,0.5],[6,2,6])
img_norm1=trans_norm1(img_tensor)
writer.add_image("T_Normalized",img_norm1,5)

#Resize
print(img.size)
trans_resize=transforms.Resize((250,250))#规定变换尺寸
img_resize=trans_resize(img)#得到PIL图片
print(img_resize.size)
img_tensor_resize=tensor_trans(img_resize)
writer.add_image("T_Resize",img_tensor_resize)

#Compose
trans_resize=transforms.Resize(200)
trans_compose=transforms.Compose([trans_resize,tensor_trans])#提供转换列表
img_resize_2=trans_compose(img)#输入trans_resize的输入
writer.add_image("T_Compose",img_resize_2,5)

#RandomCrop
trans_random=transforms.RandomCrop(200)
trans_compose_2=transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)



writer.close()


