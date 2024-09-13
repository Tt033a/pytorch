from torch.utils.data import Dataset
import cv2 #获取图片法1

from PIL import Image #获取图片法2

import os #关于系统的库
import glob

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir#实例对象的引用，构造方法    #root_dir:每张图片的地址，一般设未上一级的相对地址
        self.label_dir=label_dir  #label_dir:每张图片的标签
        self.path=os.path.join(self.root_dir,self.label_dir)#path=os.path.join(root_dir,label_dir)可以把地址和标签合在一起
        self.img_path_list = os.listdir(self.path)#获取图片名称列表
        # 获取指定目录下的所有图片
        #print(glob.glob(r"/home/qiaoyunhao/*/*.png"), "\n")  # 加上r让字符串不转义


    def __getitem__(self, idx):#idx表示编号,该例中为图片地址的列表
        img_name=self.img_path_list[idx]#获得图片名称
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)#添加名称，获得图片相对路径
        img=Image.open(img_item_path)#打开图片
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path_list)

root_dir="hymenoptera_data/dataset/train"
ants_label_dir="ants"
bees_label_dir="bees"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)
train_dataset=ants_dataset+bees_dataset#数据集加在尾部（拼接）

#img,label=ants_dataset[0] 可以获取第一张图的图片和标签
