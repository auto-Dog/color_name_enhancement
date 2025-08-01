import numpy as np
import sys
import torch
import torch.nn as nn
# import colour
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

class cvdSimulateNet(nn.Module):
    ''' 将输入图像转成CVD眼中的图像'''
    def __init__(self, cvd_type='protan', cuda=False, batched_input=False) -> None:
        super().__init__()
        self.cvd_type = cvd_type
        self.cuda_flag = cuda
        self.batched_input = batched_input
        beta_L, beta_M, beta_S = (0.63, 0.32, 0.05)
        self.beta_L = beta_L
        self.beta_M = beta_M
        self.beta_S = beta_S
        # lms_to_xyz_mat = np.array([[1.93986443, 1.34664359, 0.43044935, ],
        #                 [0.69283932, 0.34967567, 0, ],
        #                 [0, 0, 2.14687945]])    # 2006 10 degree
        # xyz_to_lms_mat = np.linalg.inv(lms_to_xyz_mat)
        # print(xyz_to_lms_mat)   # debug
        # xyz_to_lms_mat = (np.diag([beta_L,beta_M,beta_S]))@ xyz_to_lms_mat  # 根据视锥细胞比例更新变换矩阵
        xyz_to_lms_mat = np.array([[0.34*beta_L, 0.69*beta_L, -0.076*beta_L ],
                        [-0.39*beta_M, 1.17*beta_M, 0.049*beta_M ],
                        [0.038*beta_S, -0.043*beta_S, 0.48*beta_S]])
        self.xyz_to_lms_mat = torch.from_numpy(xyz_to_lms_mat).float()
        rgb_to_xyz_mat = np.array([[0.4124,0.3576,0.1805],
                                   [0.2126,0.7152,0.0722],
                                   [0.0193,0.1192,0.9505]])   # BT 709.2
        self.rgb_to_xyz_mat = torch.from_numpy(rgb_to_xyz_mat).float()

    def einsum_dot_tensor(self,batched_image,matrix):    # input BCHW
        return torch.einsum('vi,biju->bvju',matrix,batched_image)
    
    def sRGB_to_alms(self,image_sRGB:torch.tensor):
        if self.cuda_flag:
            mask_srgb = (image_sRGB<=0.04045)
            mask_srgb= mask_srgb.cuda()
            self.xyz_to_lms_mat=self.xyz_to_lms_mat.cuda()
            self.rgb_to_xyz_mat=self.rgb_to_xyz_mat.cuda()
            linear_RGB = torch.zeros_like(image_sRGB).cuda()
            image_xyz = torch.zeros_like(image_sRGB).cuda()
            image_alms = torch.zeros_like(image_sRGB).cuda()
            linear_RGB[mask_srgb] = image_sRGB[mask_srgb]/12.92
            linear_RGB[~mask_srgb] = torch.pow((image_sRGB[~mask_srgb]+0.055)/1.055,torch.tensor(2.4).cuda())# decode to linear
        else:
            mask_srgb = (image_sRGB<=0.04045)
            linear_RGB = torch.zeros_like(image_sRGB)
            image_xyz = torch.zeros_like(image_sRGB)
            image_alms = torch.zeros_like(image_sRGB)
            linear_RGB[mask_srgb] = image_sRGB[mask_srgb]/12.92
            linear_RGB[~mask_srgb] = torch.pow((image_sRGB[~mask_srgb]+0.055)/1.055,torch.tensor(2.4))# decode to linear
        # print(f'Shape {linear_RGB.shape}')   # debug
        image_xyz = self.einsum_dot_tensor(linear_RGB,self.rgb_to_xyz_mat,)
        image_alms = self.einsum_dot_tensor(image_xyz,self.xyz_to_lms_mat,)
        return image_alms
    
    def lRGB_to_alms(self,linear_RGB:torch.tensor):
        # print(f'Shape {linear_RGB.shape}')   # debug
        image_xyz = self.einsum_dot_tensor(linear_RGB,self.rgb_to_xyz_mat,)
        image_alms = self.einsum_dot_tensor(image_xyz,self.xyz_to_lms_mat,)
        return image_alms


    # def quantify_alms(self,image_alms:torch.tensor):
    #     ''' image_alms：视锥细胞响应，值域0-1.  
    #     假设环境光最大光子数1e5，等步长量化，即采用aLMS均为1e4点的色差椭球，其轴长为1*1e2/2e5'''
    #     channel_max=(0.6,0.6,0.025)
    #     bins=256
    #     quantify_out = image_alms
    #     quantify_out[:,0,:,:] = image_alms[:,0,:,:]/channel_max[0]
    #     quantify_out[:,1,:,:] = image_alms[:,1,:,:]/channel_max[1]
    #     quantify_out[:,2,:,:] = image_alms[:,2,:,:]/channel_max[2]*bins
    #     ori_noise /= bins
    #     out = input+ori_noise
    #     return out

    def add_noise(self,image_alms:torch.tensor,Y_max=1e4):
        ''' image_alms：视锥细胞响应，值域0-1. 噪声阈值为取值的sqrt倍. 
        Fonseca, María da, and Inés Samengo. 
        Derivation of Human Chromatic Discrimination Ability from an Information-Theoretical Notion of Distance in Color Space.
        https://doi.org/10.1162/NECO_A_00903.
        '''

        # # 即使加了高斯噪声，模型也可以从局部均值求解得到某颜色的精确值。因此，需要量化，让CVD和正常人的LMS空间分辨率一样。
        ref_threshold = torch.tensor([0.586,0.293,0.025]).float().to(image_alms.device)
        ref_threshold = torch.sqrt(ref_threshold* Y_max)/ Y_max  # 参考阈值
        if len(image_alms.shape)==4:
            ref_threshold = ref_threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif len(image_alms.shape)==3:
            ref_threshold = ref_threshold.unsqueeze(-1).unsqueeze(-1)
        elif len(image_alms.shape)==2:
            ref_threshold = ref_threshold.unsqueeze(0)
        # 量化方案
        image_alms_denormalized = torch.abs(image_alms*Y_max)
        image_alms_denormalized = torch.round(2*torch.sqrt(image_alms_denormalized)) # 对y=2\sqrt(x)均匀量化后回复，等效JND=\sqrt(x)
        image_alms_denormalized = (image_alms_denormalized**2)/4

        noise_std = torch.sqrt(image_alms_denormalized)
        noise = torch.randn_like(image_alms_denormalized) * noise_std
        

        noisy_image = image_alms_denormalized + noise
        # print('Noise stds:', noise/Y_max)  # debug
        image_alms = noisy_image / Y_max  # 归一化回0-1范围
        return image_alms.clamp(0, 1)  # 确保值域在0-1之间

    def forward(self,image_sRGB):
        ''' image RGB: 未经处理的原始图像，值域0-1'''
        if not self.batched_input:
            image_sRGB = image_sRGB.unsqueeze(0)
        lms_image = self.sRGB_to_alms(image_sRGB)
        lms_image_cvd = lms_image
        if self.cvd_type == 'protan':
            lms_image_cvd[:,0,:,:] = lms_image_cvd[:,1,:,:]/self.beta_M*self.beta_L # 缺L
        elif self.cvd_type == 'deutan':
            lms_image_cvd[:,1,:,:] = lms_image_cvd[:,0,:,:]/self.beta_L*self.beta_M # 缺M
        elif 'protan_' in self.cvd_type:   # protan_xxx形式
            degree = int(self.cvd_type.split('_')[1])
            lms_image_cvd[:,0,:,:] = degree/100.*lms_image_cvd[:,1,:,:]/self.beta_M*self.beta_L +\
                                    (1-degree/100.)*lms_image_cvd[:,0,:,:] # 部分缺L
        elif 'deutan_' in self.cvd_type:   # deutan_xxx形式
            degree = int(self.cvd_type.split('_')[1])
            lms_image_cvd[:,1,:,:] = degree/100.*lms_image_cvd[:,0,:,:]/self.beta_L*self.beta_M +\
                                    (1-degree/100.)*lms_image_cvd[:,1,:,:] # 部分缺M
        if not self.batched_input:
            lms_image_cvd = lms_image_cvd.squeeze(0)
        # 添加噪声，模拟人眼对颜色的感知 y=Hx+n. 由于加噪声步骤包含不可微操作，故进行STE
        lms_image_cvd = (self.add_noise(lms_image_cvd) - lms_image_cvd).detach()+lms_image_cvd 
        return lms_image_cvd
    
if __name__ == '__main__':
    myobserver = cvdSimulateNet(cvd_type='protan_90',cuda=False,batched_input=True)
    # image_sample_ori = Image.open('../../../test3_o.png').convert('RGB')
    # image_sample_ori = torch.tensor(np.array(image_sample_ori)).permute(2,0,1).unsqueeze(0)/255.
    image_sample_ori = torch.tensor(np.array([[[0.35003133, 0.83144364, 0.6597791], # deutan
                                            [0.35003133, 0.83144364 ,0.6597791 ],
                                            [0.58516089 ,0.77587424 ,0.66080132],
                                            [0.73700171 ,0.71454281 ,0.6618215 ],
                                            [0.85619809 ,0.64546611 ,0.66283965],
                                            [0.95678832 ,0.56522484 ,0.66385579],
                                            [0.95678832 ,0.66522484 ,0.76385579],
                                               [1.0,1.0,1.0]]])).permute(2,0,1).unsqueeze(0).float()
    image_sample_ori = torch.tensor(np.array([[
                                            [51,218,167],    # 0  # protan
                                            [200,208,165],   # 8
                                            [210,207,165],   # 9
                                            [220,205,164],   #10
                                           [229,204,164],    #11
                                           [238,203,164],
                                           [246,202,163],
                                           [254,200,163],]])).permute(2,0,1).unsqueeze(0).float()/255.
    # image_sample_ori = torch.tensor(np.array([[
    #                                         [0,50,0],    # 0
    #                                         [0,100,0],   # 8
    #                                         [0,150,0],   # 9
    #                                         [50,0,0],
    #                                         [100,0,0],   #10
    #                                        [150,0,0],    #11
    #                                 ]])).permute(2,0,1).unsqueeze(0).float()/255.

    image_sample = image_sample_ori.clone()
    print(image_sample.is_leaf)
    # Test loss backward ability
    image_sample.requires_grad = True
    # image_sample = image_sample.cuda()
    image_sample_o = myobserver(image_sample)
    # image_loss = torch.mean(torch.exp(-image_sample_o))
    # image_loss.backward()
    # print(image_sample.grad)

    image_array = image_sample_o.squeeze(0).permute(1,2,0).detach().numpy()
    print(image_array)  # debug
    # print(np.min(image_array),np.max(image_array))  # debug
    image_array = image_array/np.array([0.58,0.26,0.024])   # 归一化，方便查看
    plt.imshow(image_array)
    plt.show()  # 如果看到一只鹿，证明变换有效
