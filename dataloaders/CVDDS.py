import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,ImageNet,ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import os
from typing import Any, Callable, Optional, Tuple
# import sys
# sys.path.append("..")   # debug
from utils.cvdObserver import cvdSimulateNet
from utils.colorNamer import PLSAColorClassifier,ChipColorClassifier
from PIL import Image
import os
import pandas as pd
import json
import colour

class CVDcifar(CIFAR10):
    def __init__(        
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        patch_size = 16,
        img_size = 512,
        cvd = 'deutan'
    ) -> None:
        super().__init__(root,train,transform,target_transform,download)

        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize(self.image_size),
            ]
        )
        self.cvd_observer = cvdSimulateNet(cvd)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index] # names from CIFAR10

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.my_transform(img)
        img_target = img.clone()
        random_index = torch.randint(0,self.image_size//self.patch_size,size=(2,))
        patch_target = img[:, random_index[0]*self.patch_size:random_index[0]*self.patch_size+self.patch_size, 
                           random_index[1]*self.patch_size:random_index[1]*self.patch_size+self.patch_size]
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target # CVD image, CVD patch, image target, patch target
    
class CVDImageNet():
    def __init__(self, root: str, split: str = "train", patch_size=16, img_size=512, cvd='deutan',**kwargs: Any) -> None:
        self.root = root
        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_size,self.image_size)),
            ]
        )
        self.cvd_observer = cvdSimulateNet(cvd)
        self.color_name_embeddings = pd.read_csv('basic_color_embeddings.csv',index_col='Name')
        self.category_map = {
            'Red': 0,
            'Green': 1,
            'Blue': 2,
            'Black':3,
            'White':4,
            'Gray' :5,
            'Pink' :6,
            'Orange':7,
            'Purple':8,
            'Cyan':9,
            'Yellow':10,
            'Brown': 11
        }
        self.category_names = list(self.category_map.keys())
        if 'train' in split:
            split = 'train'
        elif 'val' in split:
            split = 'val'
        try:
            self.data_label_df = pd.read_csv('dataloaders/'+split+'_label.csv',index_col=0)
        except:
            raise IOError('Label file does not exist. Run dataloaders/make_data.py')


    def __len__(self):
        return len(self.data_label_df)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (CVD image, CVD patch, image target (original color), patch target(original color), patch color name, patch id) 
        """
        [path, patch_id, target] = self.data_label_df.loc[index].to_list()  # names form ImageNet -> ImageFolder -> DatasetFolder
        sample = self.loader(os.path.join(self.root,path))

        img = self.my_transform(sample)
        img_target = img.clone()
        patch_h = self.image_size//self.patch_size  # 32, currently, h=w
        patch_target = img[:, 
                           (patch_id//patch_h)*self.patch_size:(patch_id//patch_h+1)*self.patch_size, 
                           (patch_id%patch_h)*self.patch_size:(patch_id%patch_h+1)*self.patch_size]
        patch_color_embedding,patch_color_name = self.getEmbedding(target) # get color names   # debug
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)
        patch_id = torch.tensor(patch_id,dtype=torch.long)
        # CVD image, CVD patch, image target (original color), patch target(original color), patch color name, patch id
        return img, patch, img_target, patch_target, patch_color_name, patch_id 
    
    def getEmbedding(self,target):
        '''Given a patch's color type number, return embedding and name'''
        color_name = self.category_names[target]
        color_embedding = self.color_name_embeddings.loc[color_name].to_numpy()
        color_embedding = torch.from_numpy(color_embedding)
        # color_index = torch.tensor(color_index,dtype=torch.long)
        return color_embedding, color_name
    
    def loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class CVDImageNetTrain(CVDImageNet):
    def __init__(self, root: str, split: str = "train", patch_size=16, img_size=512, cvd='deutan',**kwargs: Any) -> None:
        super().__init__(root, split, patch_size, img_size, cvd,**kwargs)
        # 对数据集进行文件名分组、打乱
        self.grouped_data = self.data_label_df.groupby('Path').agg(list).reset_index()
        self.grouped_data = self.grouped_data.sample(frac=1).reset_index(drop=True)
        self.max_group_len = max([len(patch_ids) for patch_ids in self.grouped_data['Patch_ID']])  # 最大的组长度,用于batch对齐
        self.data_label_df = self.grouped_data

    def __len__(self):
        return len(self.data_label_df)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (CVD img, original color img_target, all_patch_color_names where masked element are color names, all_patch_mask)
        """

        [path, patch_ids, targets] = self.data_label_df.loc[index].to_list()  # names form ImageNet -> ImageFolder -> DatasetFolder
        sample = self.loader(os.path.join(self.root,path))

        img = self.my_transform(sample)
        img_target = img.clone()

        patch_h = self.image_size // self.patch_size
        total_patches = patch_h * patch_h
        # 初始化一维的 all_patch_color_names 和 all_patch_mask
        all_patch_color_names = ["None"] * total_patches
        all_patch_mask = torch.zeros(total_patches, dtype=torch.long)

        for patch_id, target in zip(patch_ids, targets):
            _, patch_color_name = self.getEmbedding(target)
            all_patch_color_names[patch_id] = patch_color_name
            all_patch_mask[patch_id] = 1
        img_cvd = self.cvd_observer(img)
        return img_cvd, img_target, all_patch_color_names, all_patch_mask


def sRGB_to_Lab(rgb1):
    rgb_batch = np.float32(rgb1)
    # 重新调整输入数组的形状，使其成为 (n, 1, 3)，符合OpenCV的要求
    ori_shape = rgb_batch.shape
    rgb_batch = rgb_batch.reshape(-1, 1, 3)
    # 使用OpenCV的cvtColor函数转换RGB到Lab
    lab_batch = cv2.cvtColor(rgb_batch, cv2.COLOR_RGB2Lab)
    return lab_batch.reshape(ori_shape)  # 还原形状

class CVDImageNetRand(ImageFolder):
    def __init__(self, root: str, split: str = "train", patch_size=16, img_size=512, cvd='deutan',**kwargs: Any) -> None:
        target_path = os.path.join(root,split)
        super().__init__(target_path, **kwargs)
        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_size,self.image_size)),
            ]
        )
        self.cvd_observer = cvdSimulateNet(cvd)
        self.color_name_embeddings = pd.read_csv('basic_color_embeddings.csv',index_col='Name')
        colornamer_obj = PLSAColorClassifier('w2cM.xml')
        self.classify_color = colornamer_obj.classify_color
        # colornamer = ChipColorClassifier('../name_table.xlsx')
        # self.classify_color = colornamer.classify_color
        # df = pd.read_excel('name_table.xlsx',index_col='Colorname')  # 替换为您的文件路径
        # # 初始化字典
        # self.color_names = []
        # color_value = []
        # # 遍历DataFrame中的每一行
        # for index, row in df.iterrows():
        #     # 获取颜色分类
        #     self.color_names.append(row['Classification'])
        #     # 将RGB字符串转换为数组
        #     rgb_array = [int(x) for x in row['RGB'].split(',')]
        #     color_value.append(rgb_array)

        # self.color_value_array = np.array(color_value)
        # self.color_value_array_lab = sRGB_to_Lab(self.color_value_array/255.)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]  # names form ImageNet -> ImageFolder -> DatasetFolder
        sample = self.loader(path)

        img = self.my_transform(sample)
        img_target = img.clone()
        random_index = torch.randint(0,self.image_size//self.patch_size,size=(2,))
        patch_id = (random_index[0] * (self.image_size // self.patch_size)) + random_index[1]
        patch_id = patch_id.long()
        patch_target = img[:, random_index[0]*self.patch_size:random_index[0]*self.patch_size+self.patch_size, 
                           random_index[1]*self.patch_size:random_index[1]*self.patch_size+self.patch_size]
        patch_color_embedding,patch_color_name = self.getEmbedding(patch_target) # get color names   # debug
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target, patch_color_name, patch_id # CVD image, CVD patch, image target, patch target
    
    def getEmbedding(self,color_patch):
        '''Given a color patch, return its color type number and embedding'''
        color_patch_mean = torch.mean(color_patch,dim=[1,2])*255
        color_name,color_index = self.classify_color(color_patch_mean)
        color_embedding = self.color_name_embeddings.loc[color_name].to_numpy()
        color_embedding = torch.from_numpy(color_embedding)
        # color_index = torch.tensor(color_index,dtype=torch.long)
        return color_embedding, color_name
    
    # def classify_color(self,rgb):
    #     ''' Given a color RGB, return its color type number '''
    #     rgb = rgb.numpy()  # RGB tensor to numpy
    #     # calculate norm as distance between input color and template colors
    #     category_map = {
    #         'Red': 0,
    #         'Green': 1,
    #         'Blue': 2,
    #         'Black':3,
    #         'White':4,
    #         'Gray' :5,
    #         'Pink' :6,
    #         'Orange':7,
    #         'Purple':8,
    #         'Cyan':9,
    #         'Yellow':10,
    #         'Brown': 11
    #     }
    #     # distances = np.linalg.norm(self.color_value_array - rgb, axis=1)
    #     # index = np.argmin(distances)
    #     # return self.color_names[index], category_map[self.color_names[index]]   # return color words and index

    #     input_lab = sRGB_to_Lab(rgb/255.)
    #     distances = np.linalg.norm(self.color_value_array_lab - input_lab, axis=1)

    #     index = np.argmin(distances)
    #     # check if it is gray
    #     if(input_lab[0]>10 and input_lab[0]<90 and abs(input_lab[1])<5 and abs(input_lab[2])<5):
    #         return 'Gray',5
    #     # return color_name[index],None
    #     return self.color_names[index],category_map[self.color_names[index]]

class CVDPlace(CVDImageNet):
    def __init__(self, root: str, split: str = "train", patch_size=4, img_size=64,**kwargs: Any) -> None:
        super().__init__(root, split, patch_size, img_size, cvd = 'deutan',**kwargs)

if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    CVD_set = CVDImageNet('/work/mingjundu/imagenet100k/',split='imagenet_subtrain')
    CVD_loader = torch.utils.data.DataLoader(CVD_set,batch_size=10,shuffle = True)
    for outputs in CVD_loader:
        img_ori = outputs[2]
        patch_ori = outputs[3]
        patch_ori_name = outputs[4]
        patch_ori_emb = outputs[5]
        show = ToPILImage()
        show(img_ori[0]).save('img_ori.png')
        show(patch_ori[0]).save('patch_ori.png')
        print('color name:',patch_ori_name)
        break
    

