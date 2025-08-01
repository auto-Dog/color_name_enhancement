import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib
matplotlib.use('Agg')  # 设置 Matplotlib 后端为 Agg
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import Logger
from tqdm import tqdm
from dataloaders.CVDDS import CVDcifar,CVDImageNet,CVDImageNetTrain,CVDPlace,CVDImageNetRand
from network import ViT,colorLoss, colorFilter, SSIMLoss, colorLossEnhance
from utils.cvdObserver import cvdSimulateNet
from utils.conditionP import conditionP
from utils.utility import patch_split,patch_compose
from kornia.color import rgb_to_lab
# hugface官方实现
# from transformers import ViTImageProcessor, ViTForImageClassification
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits

dataset = 'local'
num_classes = 6

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=8)
parser.add_argument('--size',type=int, default=256)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/data/mingjundu/imagenet100k/')
parser.add_argument('--test_split', type=str, default='imagenet_subval')
parser.add_argument("--cvd", type=str, default='deutan')
parser.add_argument("--tau", type=float, default=0.3)
parser.add_argument("--x_bins", type=float, default=128.0)  # noise setting, to make input continues-like
parser.add_argument("--cvd_warmup_iter", type=float, default=2000.)
parser.add_argument("--prefix", type=str, default='vit_cn6b')
parser.add_argument('--from_check_point',type=str,default='')
parser.add_argument('--train_mode',type=str,default='est',choices=['est','optim','both'])  # est: 只训练颜色估计模块；optim: 只训练颜色增强模块；both: 两个模块都训练
args = parser.parse_args()

print(args) # show all parameters
### write model configs here
save_root = './run/'+args.prefix
pth_location = './Models/model_'+args.prefix+'.pth'
pth_optim_location = './Models/model_'+args.prefix+'_optim_base'+'.pth'
# pth_optim_location = './Models/model_vit_cn7aE_D100_optim_base'+'.pth'
ckp_location = './Models/'+args.from_check_point
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
train_val_percent = 0.8
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# skf = StratifiedGroupKFold(n_splits=n_splits)

trainset = CVDImageNetTrain(args.dataset,split='imagenet_subtrain',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
valset = CVDImageNet(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # cvd模拟器
lab_normalize = transforms.Compose([transforms.Normalize((0, 0, 0), (100, 128, 128))])  # lab归一化
rgb_to_lab_normal = lambda x: lab_normalize(rgb_to_lab(x))
# train_size = int(len(trainset) * train_val_percent)   # not suitable for ImageNet subset
# val_size = len(trainset) - train_size
# trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True,num_workers=16)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True,num_workers=16)
# testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)
# inferenceloader = torch.utils.data.DataLoader(inferenceset,batch_size=args.batchsize,shuffle = False,)
# trainval_loader = {'train' : trainloader, 'valid' : validloader}

model = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()
optim_model = colorFilter().cuda()
optim_model = nn.DataParallel(optim_model,device_ids=list(range(torch.cuda.device_count())))
optim_model = optim_model.cuda()

criterion = colorLoss(args.tau)
criterion2 = SSIMLoss()
# criterion3 = colorLossEnhance(args.tau)
# criterion3 = nn.MSELoss()    # debug
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

# Update 11.15
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
optimizer_optim = torch.optim.Adam(optim_model.parameters(), lr=args.lr, weight_decay=5e-5)
lr_lambda = lambda epoch: min(1.0, (epoch + 1)/5.)  # noqa
lrsch = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
iter_count = 0
# lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=0.3)
logger.auto_backup('./')

import time
def train(trainloader, model, criterion, optimizer, lrsch, logger, args, phase='train', optim_model=None):
    global iter_count
    train_iter = tqdm(trainloader,ascii=True,ncols=60)
    if phase=='train':
        model.train()
        logger.update_step()
        loss_logger = 0.
        label_list = []
        pred_list  = []
        for img, img_target, all_patch_color_names, all_patch_mask in train_iter:
            optimizer.zero_grad()
            img = img.cuda()
            # st_time = time.time()   # debug
            # print('Timer now')  # debug
            outs = model(img)   
            # outs = model(add_input_noise(img,bins=args.x_bins))

            batch_size = img.size(0)
            bxn_embeddings = []
            bxn_colors_list = []
            all_patch_color_names_T = list(map(list, zip(*all_patch_color_names)))
            # 每个batch内b个图，每个图有n个（不固定）选中patch。对应了n个embedding和n个颜色名称
            for b in range(batch_size):
                # 展平掩码和颜色名称列表
                flat_mask = all_patch_mask[b].flatten()
                flat_color_names = all_patch_color_names_T[b]
                # 获取有效位置的索引
                valid_indices = flat_mask.nonzero(as_tuple=True)[0]
                # 提取有效位置的嵌入
                valid_embeddings = outs[b,valid_indices]
                bxn_embeddings.append(valid_embeddings)
                # 提取有效颜色名称并转换为元组
                valid_color_names = tuple([flat_color_names[i] for i in valid_indices])
                bxn_colors_list.append(valid_color_names)

            # 拼接嵌入
            bxn_embeddings = torch.cat(bxn_embeddings, dim=0)
            bxn_colors = ()
            for sub_tuple in bxn_colors_list:
                bxn_colors += sub_tuple
            # # debug
            # print("\n bxn_colors:",bxn_colors)
            # print("\n bxn_embeddings size:",bxn_embeddings.size())
            loss_batch = criterion(bxn_embeddings, bxn_colors)
            # print('Loss Func. use:',time.time()-st_time)    # debug
            pred, label = criterion.classification(bxn_embeddings, bxn_colors)
            # print('Classification use:',time.time()-st_time)    # debug
            label_list.extend(label.cpu().detach().tolist())
            pred_list.extend(pred.cpu().detach().tolist())
            # img_target = img_target.cuda()
            # print("opt tensor:",out)
            # ci_rgb = ci_rgb.cuda()

            # if epoch>30:
            #     # 冻结部分层
            #     for name, param in model.named_parameters():
            #         if ("transformer" in name):
            #             param.requires_grad = False
            # loss_batch = criterion(outs,img_target)
            loss_batch.backward()
            loss_logger += loss_batch.item()    # 显示全部loss
            optimizer.step()
            train_iter.set_postfix(loss=loss_batch.item())
        lrsch.step()

        loss_logger /= len(trainloader)
        print("Train loss:",loss_logger)
        log_metric('Train',logger,loss_logger,label_list,pred_list)
        if not (logger.global_step % args.save_interval):
            logger.save(model,optimizer, lrsch, criterion)
    
    if phase=='optim':
        model.eval()
        optim_model.train()
        loss_logger = 0.
        label_list = []
        pred_list  = []
        
        for img, img_target, all_patch_color_names, all_patch_mask in train_iter:
            optimizer.zero_grad()
            img_target = img_target.cuda()
            img_opt = optim_model(img_target)
            img_opt_cvd = cvd_process(img_opt)
            outs = model(img_opt_cvd)
            # 将多个image的结果拼接到一起
            batch_size = img.size(0)
            bxn_embeddings = []
            bxn_colors_list = []
            all_patch_color_names_T = list(map(list, zip(*all_patch_color_names)))
            # 每个batch内b个图，每个图有n个（不固定）选中patch。对应了n个embedding和n个颜色名称
            for b in range(batch_size):
                # 展平掩码和颜色名称列表
                flat_mask = all_patch_mask[b].flatten()
                flat_color_names = all_patch_color_names_T[b]

                # 获取有效位置的索引
                valid_indices = flat_mask.nonzero(as_tuple=True)[0]

                # 提取有效位置的嵌入
                valid_embeddings = outs[b,valid_indices]
                bxn_embeddings.append(valid_embeddings)

                # 提取有效颜色名称并转换为元组
                valid_color_names = tuple([flat_color_names[i] for i in valid_indices])
                bxn_colors_list.append(valid_color_names)

            # 拼接嵌入
            bxn_embeddings = torch.cat(bxn_embeddings, dim=0)
            bxn_colors = ()
            for sub_tuple in bxn_colors_list:
                bxn_colors += sub_tuple

            loss_batch = 0.6*min(1.0,(iter_count/args.cvd_warmup_iter))*criterion(bxn_embeddings,bxn_colors)\
                +0.4*criterion2(img_opt,img_target)
            # print('Loss Func. use:',time.time()-st_time)    # debug
            pred, label = criterion.classification(bxn_embeddings, bxn_colors)
            label_list.extend(label.cpu().detach().tolist())
            pred_list.extend(pred.cpu().detach().tolist())
            loss_batch.backward()
            # 查看梯度
            # print(optim_model.module.outc.conv.weight.grad.max())
            loss_logger += loss_batch.item()    # 显示全部loss
            optimizer.step()
            if iter_count%1000==0:
                sample_enhancement(model,None,iter_count,args)
            iter_count += 1
            # 在 tqdm 进度条中实时显示当前批次的 loss
            train_iter.set_postfix(loss=loss_batch.item())

        # lrsch.step()

        loss_logger /= len(trainloader)
        print("Train Optim loss:",loss_logger)
        log_metric('Train Optim',logger,loss_logger,label_list,pred_list)
        

def validate(testloader, model, criterion, optimizer, lrsch, logger, args, phase='eval', optim_model=None):
    model.eval()
    optim_model.eval()
    loss_logger = 0.
    label_list = []
    pred_list  = []
    val_iter = tqdm(testloader,ascii=True,ncols=60)
    for img, ci_patch, img_ori, patch_ori, patch_color_name, patch_id in val_iter:
        if phase == 'eval':
            with torch.no_grad():
                outs = model(img.cuda()) 
        elif phase == 'optim':
            with torch.no_grad():
                img_ori = img_ori.cuda()
                img_opt = optim_model(img_ori)
                img_opt_cvd = cvd_process(img_opt)
                outs = model(img_opt_cvd) 
        # ci_rgb = ci_rgb.cuda()
        # img_target = img_target.cuda()
        # print("label:",label)
        batch_index = torch.arange(len(outs),dtype=torch.long)   # 配合第二维度索引使用
        outs = outs[batch_index,patch_id] # 取出目标位置的颜色embedding
        loss_batch = criterion(outs,patch_color_name)
        loss_logger += loss_batch.item()    # 显示全部loss
        pred,label = criterion.classification(outs,patch_color_name)
        label_list.extend(label.cpu().detach().tolist())
        pred_list.extend(pred.cpu().detach().tolist())
        # 在 tqdm 进度条中实时显示当前批次的 loss
        val_iter.set_postfix(loss=loss_batch.item())
    loss_logger /= len(testloader)
    if phase == 'eval':
        print("Val loss:",loss_logger)
        acc = log_metric('Val', logger,loss_logger,label_list,pred_list)
        return acc, model.state_dict()
    elif phase == 'optim':
        print("Val Optim loss:",loss_logger)
        acc = log_metric('Val Optim', logger,loss_logger,label_list,pred_list)
        return acc, optim_model.state_dict()
    
def sample_enhancement(model,inferenceloader,epoch,args):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    '''
    model.eval()
    cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # cvd模拟器，保证在同一个设备上进行全部运算
    # temploader =  CVDImageNetRand(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)   # 只利用其中的颜色命名模块
    image_sample = Image.open('images/apple.png').convert('RGB')
    # image_sample_big = np.array(image_sample)/255.   # 缓存大图
    image_sample = image_sample.resize((args.size,args.size))
    image_sample = np.array(image_sample)
    # patch_names = []
    # for patch_y_i in range(args.size//args.patch):
    #     for patch_x_i in range(args.size//args.patch):
    #         y_end = patch_y_i*args.patch+args.patch
    #         x_end = patch_x_i*args.patch+args.patch
    #         single_patch = image_sample[patch_y_i*16:y_end,patch_x_i*16:x_end,:]
    #         # calculate color names
    #         patch_rgb = np.mean(single_patch,axis=(0,1))
    #         patch_color_name,_ = temploader.classify_color(torch.tensor(patch_rgb)) # classify_color接收tensor输入
    #         patch_names.append(patch_color_name)

    image_sample = torch.tensor(image_sample).permute(2,0,1).unsqueeze(0)/255.
    image_sample = image_sample.cuda()
    img_ori = image_sample.clone()

    # 一次性生成方案：
    optim_model.eval()
    img_opt = optim_model(img_ori)    # 采用cnn变换改变色彩
    # img_cvd = cvd_process(img_opt)
    # outs = model(img_cvd)
    # outs = outs[0]  # 去掉batch维度

    ori_out_array = img_ori.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    img_out_array = img_opt.clone()
    img_out_array = img_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()

    img_diff = np.abs(img_out_array - ori_out_array)*10.0 # 夸张显示色彩差异
    img_all_array = np.clip(np.hstack([ori_out_array,img_out_array,img_diff]),0.0,1.0)
    plt.imshow(img_all_array)
    plt.savefig('./run/'+f'sample_{args.prefix}_e{epoch}.png')

def log_metric(prefix, logger, loss, target, pred):
    cls_report = classification_report(target, pred, output_dict=True, zero_division=0)
    acc = accuracy_score(target, pred)
    print(cls_report)   # all class information
    # auc = roc_auc_score(target, prob)
    logger.log_scalar(prefix+'/loss',loss,print=False)
    # logger.log_scalar(prefix+'/AUC',auc,print=True)
    logger.log_scalar(prefix+'/'+'Acc', acc, print= True)
    logger.log_scalar(prefix+'/'+'Pos_precision', cls_report['weighted avg']['precision'], print=False)
    # logger.log_scalar(prefix+'/'+'Neg_precision', cls_report['0']['precision'], print= True)
    logger.log_scalar(prefix+'/'+'Pos_recall', cls_report['weighted avg']['recall'], print=False)
    # logger.log_scalar(prefix+'/'+'Neg_recall', cls_report['0']['recall'], print= True)
    logger.log_scalar(prefix+'/'+'Pos_F1', cls_report['weighted avg']['f1-score'], print=False)
    logger.log_scalar(prefix+'/loss',loss,print=False)
    return acc   # 越大越好

testing = validate
best_score = 0

if args.test == True:
    finaltestset =  CVDImageNetRand(args.dataset,split=args.test_split,patch_size=args.patch,img_size=args.size,cvd=args.cvd)
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = True,num_workers=4)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    optim_model.load_state_dict(torch.load(pth_optim_location, map_location='cpu'))
    # sample_enhancement(model,None,-1,args)  # test optimization
    testing(finaltestloader,model,criterion,optimizer,lrsch,logger,args,'optim',optim_model)    # test performance on dataset
else:
    if args.from_check_point != '':
        model.load_state_dict(torch.load(ckp_location))
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))

        if i==0:
            sample_enhancement(model,None,i,args) # debug
        if args.train_mode == 'est':
            # 只训练估计模块
            train(trainloader, model,criterion,optimizer,lrsch,logger,args,'train',optim_model)
            score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'eval',optim_model)
            if score > best_score:
                best_score = score
                torch.save(model_save, pth_location)
        elif args.train_mode == 'both':
            # 训练估计模块和增强模块
            train(trainloader, model,criterion,optimizer,lrsch,logger,args,'train',optim_model)
            score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'eval',optim_model)
            if score > best_score:
                best_score = score
                torch.save(model_save, pth_location)
            if (i+1)%5 == 0:
                train(trainloader, model,criterion,optimizer_optim,lrsch,logger,args,'optim',optim_model)
                score_optim, model_optim_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'optim',optim_model)
                sample_enhancement(model,None,i,args)
                if score_optim > score:
                    torch.save(model_optim_save, pth_optim_location)
        elif args.train_mode == 'optim':
            # 只训练颜色增强模块
            train(trainloader, model,criterion,optimizer_optim,lrsch,logger,args,'optim',optim_model)
            score_optim, model_optim_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'optim',optim_model)
            sample_enhancement(model,None,i,args)
            if score_optim > best_score:
                best_score = score_optim
                torch.save(model_optim_save, pth_optim_location)