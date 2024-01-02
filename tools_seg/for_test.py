import cv2
import os
import torch
import time
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tools_seg.utils_tool import set_seed
import pandas as pd
from past.create_dataset import Mydataset_test,test_transform
from tqdm import tqdm
from tools_seg.Miou import Pa,calculate_miou
import argparse
import torchvision.transforms.functional as tf
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='/mnt/ai2020/orton/dataset/华录杯/hua_lu_bei_shai/save_fusai/image/class_0/', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='//mnt/ai2020/orton/dataset/华录杯/hua_lu_bei_shai/save_fusai/mask/class_0_mask/', help='labels val data path.')
parser.add_argument('--csv_dir_val', '-cv', type=str,
                    default='/mnt/ai2020/orton/codes/segmentation/hospital_csv/name.csv', help='labels val data path.')
parser.add_argument('--resize', default=480, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--workers', default=16,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--warm_epoch', '-w', default=10, type=int, help='end epoch')
parser.add_argument('--end_epoch', '-e', default=100, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default= 'UnetPlusPlus_b5_no_warm', help='checkpoint path')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default='0', type=int, help='seed_num')
parser.add_argument('--devicenum', default='3', type=str, help='use devicenum')
args = parser.parse_args()
result_path = './image_color/color_dir/fusion_fpn_unet_unpp_ensemble8/'
if not os.path.exists(result_path):
    os.mkdir(result_path)
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()
device = args.device
# print('model_ok')
set_seed(seed=2021)
print('Start loading data.')
res = args.resize
df_val = pd.read_csv(args.csv_dir_val)
val_imgs,val_masks = args.imgs_val_path,args.labels_val_path
val_imgs = [''.join([val_imgs,'/',i.replace('.png','.png')]) for i in df_val['image_name']]#[0:10]
val_masks = [''.join([val_masks,'/',i]) for i in df_val['image_name']]#[0:10]
imgs_val = [cv2.resize(cv2.imread(i), (res,res))[:,:,::-1] for i in val_imgs]
masks_val = [cv2.resize(cv2.imread(i), (res,res))[:,:,0] for i in val_masks]
test_transform = test_transform

# c=int(len(imgs)*0.2)
after_read_date = time.time()
print('data_time',after_read_date-begin_time)
rot = tf.rotate
best_acc_final = []
self_ensemble = True

def infer(net,image):
    net.eval()
    with torch.no_grad():
        if self_ensemble:
            pre_1 = net(image)
            pre_2 = net(image.flip([-1])).flip([-1])
            pre_3 = net(image.flip([-2])).flip([-2])
            pre_4 = net(image.flip([-1, -2])).flip([-1, -2])
            pre5 =  rot(net(rot(image,90)),-90)
            pre6 =  rot(net((rot(image,90).flip([-1]))).flip([-1]),-90)
            pre7 =  rot(net((rot(image,90).flip([-2]))).flip([-2]),-90)
            pre8 =  rot(net((rot(image,90).flip([-1,-2]))).flip([-1,-2]),-90)
            out = pre_1 + pre_2 + pre_3 + pre_4 + pre5 + pre6 + pre7 + pre8
        else:
            out = net(image)
    return out
import time

def main():

    net =smp.Unet(encoder_name='timm-efficientnet-b3',encoder_weights=None,classes=2,decoder_attention_type='scse').to(device)
    state_dict = torch.load('/mnt/ai2020/orton/codes/segmentation/5fold95/newsegUn_scse_ns/Unetscse_b3_ns_fold_0.pth')#['net']
    # # new_state_dict = OrderedDict()
    # # for k, v in state_dict.items():
    # #     name = k[7:] # remove `module.`这里填写7或者13或者6根据实际情况module.net  vs module.model.net
    # #     new_state_dict[name] = v
    net.load_state_dict(state_dict)
    net = net.to(device)
    #
    net2 =smp.UnetPlusPlus(encoder_name='timm-efficientnet-b3',encoder_weights = None,classes=2).to(device)
    state_dict2 = torch.load('/mnt/ai2020/orton/codes/segmentation/5fold95/newsegUnpp/unetpp_b3_fold_0.pth')
    net2.load_state_dict(state_dict2)
    net2 = net2.to(device)
    #
    net3 =smp.FPN(encoder_name='timm-efficientnet-b3',encoder_weights = None,classes=2).to(device)
    state_dict3 = torch.load('/mnt/ai2020/orton/codes/segmentation/5fold95/newsegFPN_ns/FPN_b3_ns_fold_0.pth')
    net3.load_state_dict(state_dict3)
    net3 = net3.to(device)
    #
    # net4 =smp.Linknet(encoder_name='efficientnet-b5',encoder_weights = None,classes=2).to(device)
    # state_dict4 = torch.load('/mnt/ai2020/orton/codes/segmentation/checkpoint/Linknet_efficientnet_b5_0.000516/ckpt.pth')#['net']
    # new_state_dict2 = OrderedDict()
    # for k, v in state_dict4.items():
    #     name = k[7:] # remove `module.`这里填写7或者13或者6根据实际情况module.net  vs module.model.net
    #     new_state_dict2[name] = v
    # net4.load_state_dict(new_state_dict2)
    # # net4.load_state_dict(state_dict4)
    # net4 = net4.to(device)
    list = []
    test_ds = Mydataset_test(val_imgs,imgs_val, masks_val, test_transform)
    testloader = DataLoader( test_ds, batch_size=1,pin_memory=False, num_workers=args.workers)#prefetch_factor=4)
    with tqdm(total=len(testloader), ncols=120, ascii=True) as t:
        test_running_loss = 0
        val_pa_whole = 0
        val_iou_whole = 0
        test_miou = 0
        test_pa_whole = 0
        net.eval()
        net2.eval()
        net3.eval()
        i = 0
        with torch.no_grad():
            for batch_idx,(name,imgs, masks) in enumerate(testloader):
                # t.set_description("val(Epoch{}/{})".format(epoch, epochs))
                name_here = name[batch_idx].__str__()
                name_here2 = name_here.split('/')[-1]
                imgs, masks_cuda = imgs.to(device), masks.to(device)
                imgs = imgs.float()
                out1 = infer(net3,imgs)
                # begin = time.time()
                out2 = infer(net2,imgs)
                out3 = infer(net,imgs)
                # out4 = infer(net4,imgs)
                out = out1+  out2 +out3# + out4
                # out =  out2
                # predict = out.argmax(1).squeeze(0)
                # loss = criterion(out, targets)
                # print(loss.item())
                predicted = out.argmax(1)
                # end_time_tim = time.time()
                # print('time',end_time_tim-begin)
                test_miou += calculate_miou(predicted, masks_cuda, 2).item()
                pa = Pa(predicted, masks_cuda).item()
                test_pa_whole += pa
                test_pa = test_pa_whole/(batch_idx + 1)
                test_iou = test_miou/(batch_idx + 1)
                predict=predicted.squeeze(0)
                img_np = predict.cpu().numpy()  # np.array
                img_np = (img_np * 255).astype('uint8')
                img_np = cv2.resize(img_np, dsize=(640, 640))  # 恢复原图大小
                name = val_masks[i].split('/')[-1]
                cv2.imwrite(os.path.join(result_path+'/'+str(name)), img_np)

                list.append([name_here2,pa])
                t.set_postfix(loss='{:.3f}'.format(test_running_loss / (batch_idx + 1)),
                              val_pa='{:.2f}%'.format(test_pa*100),val_iou='{:.2f}%'.format(test_iou*100))
                t.update(1)
                end_time = time.time()
                i+=1
            print(end_time-begin_time)
            print('pa',test_pa,'iou',test_iou)

            # results_file = open('/mnt/ai2020/orton/codes/segmentation/hospital_csv/fusion3.csv', 'w', newline='')
            # csv_writer = csv.writer(results_file,dialect='excel')
            # for row in list:
            #     csv_writer.writerow(row)


if __name__ == '__main__':
    main()
after_net_time = time.time()
print('net_time',after_net_time-after_read_date)
print('best_acc_final',best_acc_final)
# print(np.mean(best_acc_final))
