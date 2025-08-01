import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,ImageNet,ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import os
from typing import Any, Callable, Optional, Tuple
from collections import Counter
from PIL import Image
import os
import pandas as pd
import json
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from utils.colorNamer import ChipColorClassifier, PLSAColorClassifier
import argparse
import colour

parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--dataset', type=str, default='/data/mingjundu/imagenet100k/')
parser.add_argument('--patch',type=int, default=8)
parser.add_argument('--size',type=int, default=256)
# parser.add_argument("--cvd", type=str, default='deutan')
args = parser.parse_args()
dataset_path = args.dataset
# colornamer = ChipColorClassifier('../name_table.xlsx')
# classify_color = colornamer.classify_color
colornamer = PLSAColorClassifier('w2cM.xml')
classify_color = colornamer.classify_color

class CVDImageNet(ImageFolder):
    def __init__(self, root: str, split: str = "train", patch_size=16, img_size=512,**kwargs: Any) -> None:
        target_path = os.path.join(root,split)
        super().__init__(target_path, **kwargs)
        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_size,self.image_size),antialias=False), 
                # transforms.Resize((self.image_size,self.image_size),antialias=False),   # 注意anti-alias可能过滤掉高频颜色变化
            ]
        )
        
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
        return img, path

img_size = args.size//args.patch
trainset = CVDImageNet(dataset_path,split='imagenet_subtrain',patch_size=args.patch,img_size=img_size)
valset = CVDImageNet(dataset_path,split='imagenet_subval',patch_size=args.patch,img_size=img_size)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle = True,num_workers=8)
valloader = torch.utils.data.DataLoader(valset,batch_size=1,shuffle = True,num_workers=8)

def make_data_label(loader,filename):
    X_train = []
    Y_train = []
    for img,path in tqdm(loader):
        for i in range(50):  # 每张图采样十个颜色点，实际可能不会全部采纳
            color_names = np.array([-1,-1,-1, -1,-1,-1, -1,-1,-1])
            x_index = np.random.randint(1,img_size-1)
            y_index = np.random.randint(1,img_size-1)
            surroundings = img[0, :, x_index-1:x_index+2, y_index-1:y_index+2].reshape(3,-1)
            for j in range(9):
                _,color_names[j] = classify_color(surroundings[:,j].flatten().numpy()*255)
            target = color_names[4]
            target_patch_id = x_index*img_size+y_index
            mask = (color_names==target)
            if np.sum(mask)>=5: # voting,至少有5/9个像素支持当前颜色
                X_train.append((path[0],target_patch_id))
                Y_train.append(target)
        # if len(Y_train)>1000:   # debug
        #     time.sleep(0.5)
        #     break
    # 对数据进行抽样，以保证各类别样本数平衡       
    under_sampler = RandomUnderSampler(random_state=1337)
    X_resampled, Y_resampled = under_sampler.fit_resample(X_train,Y_train)
    print('Resample result:',Counter(Y_resampled))
    all_path,all_patch_id = zip(*X_resampled)
    df_data = {'Path':all_path,'Patch_ID':all_patch_id,'Color_ID':Y_resampled}
    df_train = pd.DataFrame(data=df_data)
    df_train.to_csv(filename)

make_data_label(trainloader,'train_label.csv')
make_data_label(valloader,'val_label.csv')