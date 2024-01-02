import cv2
import os
import random
import torch
import copy
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CE_add_pretrain.cenet_hand_encode_bo import CE_Net_
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from past.create_dataset import Mydataset,for_train_transform,test_transform
from for_fit import fit

# model = CE_Net_pp(encoder_name='timm-efficientnet-b0',encoder_weights='imagenet',classes=2)
os.environ['CUDA_VISIBLE_DEVICES']='2'
begin_time = time.time()
# model =smp.UnetPlusPlus(encoder_name='mobilenet_v2',classes=2 )
print('model_ok')
def set_seed(seed=1): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(seed=2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
res = 480
savedir = '/mnt/ai2020/orton/orton_seg_new/checkpoint_coswarm/'
save_name =savedir +  'CE_Net_original_imagenet_1e_3_'+str(res)+'/'
print(save_name)
if not os.path.exists(save_name):
    os.mkdir(save_name)
BATCH_SIZE = 16
warmup_epochs = 3
train_epochs = 20

learning_rate = 1e-4

epochs = warmup_epochs + train_epochs

print('Start loading data.')
df = pd.read_csv('/mnt/ai2020/orton/orton_seg_new/data/train_label_fixed.csv')
train_imgs = '/mnt/ai2020/orton/dataset/hua_lu_fu_sai/bei_fen/train20210811/image_resize_'+str(res)
train_masks = '/mnt/ai2020/orton/dataset/hua_lu_fu_sai/bei_fen/train20210811/mask_resize_'+str(res)
all_imgs = [''.join([train_imgs,'/',i]) for i in df['image_name']]
all_masks = [''.join([train_masks,'/',i]) for i in df['image_name']]
imgs = [cv2.resize(cv2.imread(i), (res,res))[:,:,::-1] for i in all_imgs]
masks = [cv2.resize(cv2.imread(i), (res,res))[:,:,0] for i in all_masks]
train_transform = for_train_transform(res)
test_transform = test_transform

c=int(len(imgs)*0.2)
after_read_date = time.time()
print('data_time',after_read_date-begin_time)

best_acc_final = []
for i in range(5):

    print('Fold {:} start training'.format(i))
    
    # model =smp.DeepLabV3Plus(encoder_name='timm-efficientnet-b3',encoder_weights='imagenet',classes=2 )
    model =CE_Net_(3,2)
    # model = CE_Net_pp(encoder_name='timm-efficientnet-b0',encoder_weights='imagenet',classes=2)
    for para in model.encoder.parameters():
        para.requires_grad = False
    
    train_imgs = imgs[:i*c] + imgs[(i+1)*c:]
    train_masks = masks[:i*c] + masks[(i+1)*c:]
    test_imgs = imgs[i*c:(i+1)*c]
    test_masks = masks[i*c:(i+1)*c]
    train_ds = Mydataset(train_imgs, train_masks, train_transform)
    test_ds = Mydataset(test_imgs, test_masks, test_transform)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    #exp_lr_schedular = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-10)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=(epochs//warmup_epochs))
    
    train_dl = DataLoader(
                         train_ds,
                         shuffle=True,
                         batch_size=BATCH_SIZE,
                         pin_memory=False,
                         num_workers=8,
                         drop_last=True,
                         #prefetch_factor=4

    )

    test_dl = DataLoader(
                          test_ds,
                          batch_size=BATCH_SIZE,
                          pin_memory=False,
                          num_workers=8,
                          #prefetch_factor=4

    )
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(epochs):
        if epoch == warmup_epochs:
            for para in model.encoder.parameters():
                para.requires_grad = True

        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,model,train_dl,test_dl)

        if epoch_test_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = epoch_test_acc
            torch.save(best_model_wts, ''.join([save_name, '_fold_', str(i), '.pth']))
        
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        if ((epoch_test_loss >= epoch_loss) or (epoch_test_acc <= epoch_acc)) and (epoch_test_acc > 80.0):
            break
            
        torch.cuda.empty_cache()   
         
    print('Fold {:} trained successfully. Best AP:{:5f}'.format(i, best_acc))
    best_acc_final.append('{:.5}'.format(best_acc))
    fig = plt.figure(figsize=(22,8))
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='valid loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(''.join([save_name, '_fold_', str(i), '_Loss.png']), bbtrain_imgsox_inches = 'tight')

    fig = plt.figure(figsize=(22,8))
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='valid acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(''.join([save_name, '_fold_', str(i), '_Acc.png']), bbox_inches = 'tight')
    
after_net_time = time.time()
print('net_time',after_net_time-after_read_date)
print('best_acc_final',best_acc_final)
print(np.mean(best_acc_final))
print(save_name)