from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

#生成事件文件夹 logs
writer=SummaryWriter("logs")
#for i in range(100):
    #writer.add_scalar("y=2x",3*i,i)
image_path=r"D:\Happy\Project\learn_pytorch\hymenoptera_data\dataset\train\bees\17209602_fe5a5a746f.jpg"
img_PIL=Image.open(image_path)
#PIL转换为numpy
img_NP=np.array(img_PIL)
writer.add_image("train",img_NP,5,dataformats="HWC")

writer.close()