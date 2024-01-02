import os
import torch
import time
from tools_seg.utils_tool import set_seed
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
parser.add_argument('--csv_dir_train', '-ct', type=str,
                    default='/mnt/ai2020/orton/codes/segmentation/dateset/train.csv', help='labels val data path.')
parser.add_argument('--csv_dir_val', '-cv', type=str,
                    default='/mnt/ai2020/orton/codes/segmentation/dateset/val.csv', help='labels val data path.')
parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=24,type=int,help='batchsize')
parser.add_argument('--workers', default=24,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--warm_epoch', '-w', default=10, type=int, help='end epoch')
parser.add_argument('--end_epoch', '-e', default=100, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default= 'timm-efficientnet-b5', help='checkpoint path')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default='0', type=int, help='seed_num')
parser.add_argument('--devicenum', default='0,1', type=str, help='use devicenum')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()
# print('model_ok')
set_seed(seed=2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = args.device
# model_savedir = '/mnt/ai2020/orton/codes/segmentation/cut_mix/'

# f = open(model_savedir+'option'+'.txt', "a")
# f.write(str(args))
# f.close()
model_savedir = args.checkpoint + args.save_name + str(args.lr)+ str(args.batch_size)+'/'
def write_options(model_savedir,args):
    aaa = []
    aaa.append(['lr',str(args.lr)])
    aaa.append(['batch',args.batch_size])
    f = open(model_savedir+'option'+'.txt', "a")
    for option_things in aaa:
        f.write(str(option_things)+'\n')
    f.close()

write_options(model_savedir,args)


