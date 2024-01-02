import cv2
import os
import torch
import copy
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from tools_seg.utils_tool import write_options_with_test
import segmentation_models_pytorch as smp
from tools_seg.utils_tool import set_seed#,output_figure
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt
from tools_seg import Miou
from past.create_dataset import Mydataset,for_train_transform,test_transform
from tools_seg.fit_ import fit
import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='//mnt/ai2020/orton/dataset/skin_bing_zao/train_resize_512/Images/', )
parser.add_argument('--labels_train_path', '-lt', type=str,
                    default='//mnt/ai2020/orton/dataset/skin_bing_zao/train_resize_512/Annotation/',)
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='//mnt/ai2020/orton/dataset/skin_bing_zao/val_resize_512/Images/', )
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='//mnt/ai2020/orton/dataset/skin_bing_zao/val_resize_512/Annotation/', )
parser.add_argument('--imgs_test_path', '-ite', type=str,
                    default='/mnt/ai2020/orton/dataset/skin_bing_zao/test_resize_512/Images/', )
parser.add_argument('--labels_test_path', '-lte', type=str,
                    default='//mnt/ai2020/orton/dataset/skin_bing_zao/test_resize_512/Annotation/',)
parser.add_argument('--csv_dir_train', '-ct', type=str,
                    default='dateset/only_train_use/ISIC_2017_fold0train400.csv', help='labels val data path.')
parser.add_argument('--csv_dir_val', '-cv', type=str,
                    default='dateset/ISIC2017/val_label.csv', help='labels val data path.')
parser.add_argument('--csv_dir_test', '-cvt', type=str,
                    default='dateset/ISIC2017/test_label.csv', help='labels val data path.')
parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--workers', default=16,type=int,help='batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--warm_epoch', '-w', default=0, type=int, help='end epoch')
parser.add_argument('--end_epoch', '-e', default=60, type=int, help='end epoch')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/only_train_unet/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default= 'ISIC_2017_fold0train400/', help='checkpoint path')
parser.add_argument('--seed', default='0', type=int, help='seed_num')
parser.add_argument('--train_number', default='400', type=int, help='seed_num')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')#if
parser.add_argument('--model_name', default='unet', type=str, help='unet  unpp')#if
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum

begin_time = time.time()
# print('model_ok')
set_seed(seed=2021)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
device = args.device
save_name_id = args.csv_dir_train.split('/')[-1].replace('.csv','')
model_savedir = args.checkpoint + save_name_id + '/'    #+ str(args.lr)+ '_'+ str(args.batch_size)+'/'
# model_savedir = args.checkpoint + args.save_name #+ str(args.lr)+ '_'+ str(args.batch_size)+'/'
save_name =model_savedir +'ckpt'
print(model_savedir)
if not os.path.exists(model_savedir):
    os.mkdir(model_savedir)
epochs = args.warm_epoch + args.end_epoch


# print('Start loading data.')
res = args.resize
df_train = pd.read_csv(os.path.join(os.getcwd() , args.csv_dir_train))#[:args.train_number]
df_val = pd.read_csv(os.path.join(os.getcwd() , args.csv_dir_val))
df_test = pd.read_csv(os.path.join(os.getcwd() , args.csv_dir_test))
train_imgs,train_masks = args.imgs_train_path,args.labels_train_path
val_imgs,val_masks = args.imgs_val_path,args.labels_val_path
test_imgs,test_masks = args.imgs_test_path,args.labels_test_path

train_imgs = [''.join([train_imgs,'/',i+'.jpg']) for i in df_train['image_name']]
train_masks = [''.join([train_masks,'/',i+'_segmentation.png']) for i in df_train['image_name']]
val_imgs = [''.join([val_imgs,'/',i+'.jpg']) for i in df_val['image_name']]
val_masks = [''.join([val_masks,'/',i+'_segmentation.png']) for i in df_val['image_name']]
test_imgs = [''.join([test_imgs,'/',i+'.jpg']) for i in df_test['image_name']]
test_masks = [''.join([test_masks,'/',i+'_segmentation.png']) for i in df_test['image_name']]

imgs_train = [cv2.resize(cv2.imread(i), (res,res))[:,:,::-1] for i in train_imgs]
masks_train = [cv2.resize(cv2.imread(i), (res,res))[:,:,0] for i in train_masks]
imgs_val = [cv2.resize(cv2.imread(i), (res,res))[:,:,::-1] for i in val_imgs]
masks_val = [cv2.resize(cv2.imread(i), (res,res))[:,:,0] for i in val_masks]
train_transform = for_train_transform()
test_transform = test_transform

# c=int(len(imgs)*0.2)
after_read_date = time.time()
# print('data_time',after_read_date-begin_time)

best_acc_final = []

def main():
    if args.model_name == 'unet':
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=2).to(device)
    else:
        model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', classes=2).to(device)
    # model = torch.nn.DataParallel(model)
    train_ds = Mydataset(imgs_train, masks_train, train_transform)
    val_ds = Mydataset(imgs_val, masks_val, test_transform)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warm_epoch, T_mult=(epochs//args.warm_epoch))
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    train_dl = DataLoader(train_ds,shuffle=True,batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,
                         drop_last=True,prefetch_factor=4)

    val_dl = DataLoader(val_ds, batch_size=args.batch_size,pin_memory=False, num_workers=args.workers,prefetch_factor=4)
    best_iou = 0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    with tqdm(total=epochs, ncols=60, ) as t:
        for epoch in range(epochs):
            # if epoch == args.warm_epoch:
            #     for para in model.encoder.parameters():
            #         para.requires_grad = True
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch,epochs,model,train_dl,val_dl,device,criterion,optimizer,CosineLR)

            f = open(model_savedir + 'log'+'.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss'+ str(epoch_loss)+'  _val_loss'+str(epoch_val_loss)+
                    ' _train_miou'+str(epoch_iou)+' _val_iou'+str(epoch_val_iou)+   '\n')
            if epoch_val_iou > best_iou:
                # print('here')
                f.write( '\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_iou = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name, '.pth']))
            f.close()

            train_loss.append(epoch_loss)
            train_acc.append(epoch_iou)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_iou)
            # torch.cuda.empty_cache()
            t.update(1)
    # print('trained successfully. Best AP:{:5f}'.format(best_iou))
    # output_figure(train_loss,val_loss,train_acc,val_acc,model_savedir)

    test_number = len(test_imgs)
    imgs_test = [cv2.resize(cv2.imread(i), (res,res))[:,:,::-1] for i in test_imgs]
    masks_test = [cv2.resize(cv2.imread(i), (res,res))[:,:,0] for i in test_masks]
    test_ds = Mydataset(imgs_test, masks_test, test_transform)
    test_dl = DataLoader(test_ds, batch_size=1,pin_memory=False, num_workers=args.workers,)
    model.load_state_dict(torch.load(save_name+'.pth'))
    model.eval()
    test_miou ,test_Pre ,test_recall,test_F1score ,test_pa= 0,0,0,0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            predicted = out.argmax(1)

            test_miou += Miou.calculate_miou(predicted, targets, 2).item()
            test_Pre += Miou.pre(predicted, targets).item()
            test_recall += Miou.recall(predicted, targets).item()
            test_F1score+=Miou.F1score(predicted, targets).item()
            test_pa += Miou.Pa(predicted, targets).item()
    average_test_iou = test_miou/test_number
    average_test_Pre = test_Pre/test_number
    average_test_recall = test_recall/test_number
    average_test_F1score = test_F1score/test_number
    average_test_pa = test_pa/test_number
    print('test_miou=','%.4f'%average_test_iou,'test_Pre=','%.4f'%average_test_Pre,'test_recall=',
          '%.4f'%average_test_recall,'test_F1score=','%.4f'%average_test_F1score,'test_pa=','%.4f'%average_test_pa)
    write_options_with_test(model_savedir,args,best_iou,average_test_iou,average_test_Pre,average_test_recall,
                            average_test_F1score,average_test_pa)
if __name__ == '__main__':
    main()
after_net_time = time.time()
# print('net_time',after_net_time-after_read_date)
# print('best_acc_final',best_acc_final)
# print(np.mean(best_acc_final))
# print(save_name,'\n','finish')