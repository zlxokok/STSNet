import cv2
import os
import torch
import copy
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
# from tools_mine.produse_label import produce_label
from fit import fit,set_seed,write_options
from dataset.create_dataset import Mydataset,for_train_transform,test_transform
import argparse
import warnings
import torch.backends.cudnn as cudnn
from duibi.unet import UNet
from duibi.attu.attunet import AttU_Net
from models.deeplabv3.deeplabv3_model import DeepLabV3
warnings.filterwarnings("ignore")
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_list', type=str,default='/mnt/ai2022/zlx/dataset/CCM/csv/train.csv', )
parser.add_argument('--imgs_val_list', type=str,default='/mnt/ai2022/zlx/dataset/CCM/csv/val.csv', )
parser.add_argument('--batch_size', default=4,type=int,help='batchsize')
parser.add_argument('--workers', default=4,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=100, type=int, )
parser.add_argument('--num_class', '-t', default=2, type=int,)
parser.add_argument('--device', default='cuda', type=str, )
parser.add_argument('--checkpoint', type=str, default='checkpoint/', )
parser.add_argument('--name', type=str, default='resnext101_32x4d', )
parser.add_argument('--save_name', type=str, default= 'T2_Deeplabv3_resnext50_32x4d', )
parser.add_argument('--devicenum', default='0', type=str, )
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()

set_seed(seed=2021)
device = args.device
model_savedir = args.checkpoint + args.save_name.replace("resnext50_32x4d",args.name) + '/'#+'lr'+ str(args.lr)+ 'bs'+str(args.batch_size)+'/'
save_name =model_savedir +'ckpt'
print(model_savedir)
if not os.path.exists(model_savedir):os.mkdir(model_savedir)
epochs = args.warm_epoch + args.end_epoch

train_csv = pd.read_csv(args.imgs_train_list)#[:30]
val_csv = pd.read_csv(args.imgs_val_list)#[:30]

train_imgs,train_masks = train_csv['image_name'],train_csv['image_name']
val_imgs,val_masks = val_csv['image_name'],val_csv['image_name']

train_imgs = [''.join(['/mnt/ai2022/zlx/dataset/CCM/T2/25D/train/image','/',i]) for i in train_imgs]
train_masks = [''.join(['/mnt/ai2022/zlx/dataset/CCM/T2/25D/train/label','/',i]) for i in train_masks]
# train_imgs = [cv2.imread(i) for i in train_imgs]
train_imgs = [cv2.imread(i,cv2.IMREAD_UNCHANGED) for i in train_imgs]
train_masks = [cv2.imread(i,cv2.IMREAD_UNCHANGED) for i in train_masks]
val_imgs = [''.join(['/mnt/ai2022/zlx/dataset/CCM/T2/25D/val/image','/',i]) for i in val_imgs]
val_masks = [''.join(['/mnt/ai2022/zlx/dataset/CCM/T2/25D/val/label','/',i]) for i in val_masks]
val_imgs = [cv2.imread(i,cv2.IMREAD_UNCHANGED) for i in val_imgs]
val_masks = [cv2.imread(i,cv2.IMREAD_UNCHANGED) for i in val_masks]
# train_imgs = [cv2.resize(np.load(i), (args.resize,args.resize))[:,:,::-1] for i in train_imgs]
train_transform = for_train_transform()
test_transform = test_transform

best_acc_final = []
def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    # model = smp.Unet(encoder_name='resnext50_32x4d',encoder_weights=None,in_channels=3,classes=2).to(device)
    model = DeepLabV3(encoder_name=args.name ,encoder_weights=None,classes=2).to(device)

    # model.encoder.load_state_dict(torch.load('tools_seg/resnet34-333f7ec4.pth'))
    model = model.to('cuda')

    train_ds = Mydataset(train_imgs, train_masks, train_transform)
    val_ds = Mydataset(val_imgs, val_masks, test_transform)

    # criterion = smp.losses.FocalLoss(mode='multiclass').to('cuda')
    criterion = nn.CrossEntropyLoss(weight=None).to('cuda')  # weight=torch.tensor([1,10]
    # criterion2 = smp.losses.DiceLoss(mode='multiclass').to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=False, num_workers=4,
                          drop_last=True, )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=False, num_workers=8, )
    best_acc = 0
    with tqdm(total=epochs, ncols=60) as t:
        for epoch in range(epochs):
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch,epochs,model,train_dl,val_dl,device,criterion,optimizer,CosineLR)

            f = open(model_savedir + 'log'+'.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss'+ str(epoch_loss)+'  _val_loss'+str(epoch_val_loss)+
                    ' _epoch_acc'+str(epoch_iou)+' _val_iou'+str(epoch_val_iou)+   '\n')

            if epoch_val_iou > best_acc:
                f.write( '\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name,  '.pth']))
            torch.save(best_model_wts, ''.join([save_name, 'last.pth']))
            f.close()
            # torch.cuda.empty_cache()
            t.update(1)
    write_options(model_savedir,args,best_acc)


    # dice,pre,recall,f1_score ,pa = test_mertric_here(model,test_imgs,test_masks,save_name)
    # f = open('./checkpoint/result_txt/' + 'r34u_base_train'+'.txt', "a")
    # f.write(str(model_savedir)+'  dice'+str(dice)+'  pre'+str(pre)+'  recall'+str(recall)+
    #         '  f1_score'+str(f1_score)+'  pa'+str(pa)+'\n')
    # f.close()
    # print('test_acc',acc)
    # print('best_acc','%.4f'%best_acc)

if __name__ == '__main__':
    main()


# print(save_name)