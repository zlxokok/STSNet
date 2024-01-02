import cv2
import os
import torch
import copy
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tools_seg.utils_tool import set_seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from past.create_dataset import Mydataset,for_train_transform,test_transform
from tqdm import tqdm
from tools_seg.Miou import Pa

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='//mnt/ai2019/ljl/data/gastric/total/2048/orton/orton/train_image/', help='imgs train data path.')
parser.add_argument('--labels_train_path', '-lt', type=str,
                    default='//mnt/ai2019/ljl/data/gastric/total/2048/orton/orton/train_mask/', help='labels train data path.')
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='//mnt/ai2019/ljl/data/gastric/total/2048/orton/orton_add_hua_lu_bei/val_image/', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='//mnt/ai2019/ljl/data/gastric/total/2048/orton/orton_add_hua_lu_bei/val_mask/', help='labels val data path.')
parser.add_argument('--csv_dir_train', '-lv', type=str,
                    default='/mnt/ai2020/orton/codes/orton_seg_new/data/train_label_fixed.csv', help='labels val data path.')
parser.add_argument('--csv_dir_train_val', '-lv', type=str,
                    default='/mnt/ai2020/orton/codes/orton_seg_new/data/train_label_fixed.csv', help='labels val data path.')
parser.add_argument('--resize', default=640, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--workers', default=8,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--warm_epoch', '-e', default=3, type=int, help='end epoch')
parser.add_argument('--end_epoch', '-e', default=30, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default= 'FPN_b3_640_after_hospital_iv', help='checkpoint path')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default='0', type=int, help='seed_num')
parser.add_argument('--devicenum', default='2,3', type=str, help='use devicenum')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()
# print('model_ok')
set_seed(seed=2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = args.device
model_savedir = args.checkpoint + args.save_name + '/'
save_name =model_savedir +'ckpt.pth'
print(model_savedir)
if not os.path.exists(model_savedir):
    os.mkdir(model_savedir)
tb_path = args.tb_path +  args.save_name + '/'
if not os.path.exists(tb_path):
    os.mkdir(tb_path)
epochs = args.warm_epoch + args.end_epoch

print('Start loading data.')
res = args.resize
df = pd.read_csv(args.csv_dir)
train_imgs = args.imgs_train_path
train_masks = args.labels_train_path
train_imgs = [''.join([train_imgs,'/',i]) for i in df['image_name']]
train_masks = [''.join([train_masks,'/',i]) for i in df['image_name']]
imgs = [cv2.resize(cv2.imread(i), (res,res))[:,:,::-1] for i in all_imgs]
masks = [cv2.resize(cv2.imread(i), (res,res))[:,:,0] for i in all_masks]
train_transform = for_train_transform(res)
test_transform = test_transform

c=int(len(imgs)*0.2)
after_read_date = time.time()
print('data_time',after_read_date-begin_time)

from torch.cuda.amp import autocast, GradScaler


best_acc_final = []

def fit(epoch, model, trainloader, testloader,criterion,optimizer,CosineLR):
    with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
        scaler = GradScaler()
        if torch.cuda.is_available():
            model.to('cuda')
        running_loss = 0
        model.train()
        train_pa_whole = 0
        for  batch_idx, (imgs, masks) in enumerate(trainloader):
            t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            with autocast():
                masks_pred = model(imgs)
                loss = criterion(masks_pred, masks_cuda)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                predicted = masks_pred.argmax(1)
                train_pa = Pa(predicted,masks_cuda)
                train_pa_whole += train_pa.item()

                running_loss += loss.item()
            epoch_acc = train_pa_whole/(batch_idx+1)
            t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
                          train_pa='{:.2f}%'.format(epoch_acc*100))
            t.update(1)
        # epoch_acc = correct / total
        epoch_loss = running_loss / len(trainloader.dataset)
    with tqdm(total=len(testloader), ncols=120, ascii=True) as t:
        test_running_loss = 0
        val_pa_whole = 0
        model.eval()
        with torch.no_grad():
            for batch_idx,(imgs, masks) in enumerate(testloader):
                t.set_description("val(Epoch{}/{})".format(epoch, epochs))
                imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
                imgs = imgs.float()
                masks_pred = model(imgs)
                predicted = masks_pred.argmax(1)
                val_pa = Pa(predicted,masks_cuda)
                val_pa_whole += val_pa.item()
                loss = criterion(masks_pred, masks_cuda)
                test_running_loss += loss.item()
                epoch_test_acc = val_pa_whole/(batch_idx+1)
                t.set_postfix(loss='{:.3f}'.format(test_running_loss / (batch_idx + 1)),
                              val_pa='{:.2f}%'.format(epoch_test_acc*100))
                t.update(1)
        # epoch_test_acc = test_correct / test_total
        epoch_test_loss = test_running_loss / len(testloader.dataset)
        #if epoch > 2:
        CosineLR.step()
        return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


def main():

    model =smp.DeepLabV3Plus(encoder_name='timm-efficientnet-b3',encoder_weights='imagenet',classes=2 ).to(device)
    for para in model.encoder.parameters():
        para.requires_grad = False
    
    train_imgs = imgs[c:]
    train_masks =  masks[c:]
    test_imgs = imgs[:c]
    test_masks = masks[:c]
    train_ds = Mydataset(train_imgs, train_masks, train_transform)
    test_ds = Mydataset(test_imgs, test_masks, test_transform)
    
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #exp_lr_schedular = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-10)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warm_epoch, T_mult=(epochs//args.warm_epoch))
    
    train_dl = DataLoader(train_ds,shuffle=True,batch_size=args.batch_size,pin_memory=False,num_workers=args.workers,
                         drop_last=True,
                         #prefetch_factor=4
)

    test_dl = DataLoader( test_ds,  batch_size=args.batch_size,pin_memory=False, num_workers=args.workers)#prefetch_factor=4)
    
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(epochs):
        if epoch == args.warm_epoch:
            for para in model.encoder.parameters():
                para.requires_grad = True

        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,model,train_dl,test_dl,criterion,optimizer,CosineLR)

        if epoch_test_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = epoch_test_acc
            torch.save(best_model_wts, ''.join([save_name, '.pth']))
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        # if ((epoch_test_loss >= epoch_loss) or (epoch_test_acc <= epoch_acc)) and (epoch_test_acc > 80.0):
        #     break
            
        torch.cuda.empty_cache()   
         
    print('trained successfully. Best AP:{:5f}'.format(best_acc))
    best_acc_final.append('{:.5}'.format(best_acc))
    fig = plt.figure(figsize=(22,8))
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='valid loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(''.join([save_name, '_Loss.png']), bbtrain_imgsox_inches = 'tight')

    fig = plt.figure(figsize=(22,8))
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='valid acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(''.join([save_name, '_Acc.png']), bbox_inches = 'tight')
    
after_net_time = time.time()
print('net_time',after_net_time-after_read_date)
print('best_acc_final',best_acc_final)
print(np.mean(best_acc_final))
print(save_name)