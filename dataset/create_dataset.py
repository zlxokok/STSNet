import torch
from torch.utils import data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import shutil
import random

class Mydataset_no_read_class(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        labels = int(label)
        data = self.transforms(image=img)

        return data['image'], labels

    def __len__(self):
        return len(self.imgs)




class Mydataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        img = img/1.0
        img = np.expand_dims(img,2)
        img = np.tile(img, 3)
        label = (self.labels[index]).astype('uint8')
        label[label>0]=1
        data = self.transforms(image=img,mask=label)
        # labelll = torch.max(data['mask'])z
        return data['image'],(data['mask']).long()

    def __len__(self):
        return len(self.imgs)
#
#
#
# class Mydataset(data.Dataset):
#     def __init__(self, img_paths, labels, transform):
#         self.imgs = img_paths
#         self.labels = labels
#         self.transforms = transform
#
#     def augment_data(self, img, label):
#         augmented_imgs = []
#         augmented_labels = []
#
#         img_f1 = cv2.flip(img, 0).astype('uint8')
#         label_f1 = cv2.flip(label, 0).astype('uint8')
#         augmented_imgs.append(img_f1)
#         augmented_labels.append(label_f1)
#
#         img_f2 = cv2.flip(img, 1).astype('uint8')
#         label_f2 = cv2.flip(label, 1).astype('uint8')
#         augmented_imgs.append(img_f2)
#         augmented_labels.append(label_f2)
#
#         img_f3 = img[10:470, 10:470]
#         label_f3 = label[10:470, 10:470]
#         img_f3 = cv2.resize(img_f3, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f3 = cv2.resize(label_f3, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         augmented_imgs.append(img_f3)
#         augmented_labels.append(label_f3)
#
#         img_f4 = cv2.flip(img, 1).astype('uint8')
#         label_f4 = cv2.flip(label, 1).astype('uint8')
#         img_f4 = img_f4[10:470, 10:470]
#         label_f4 = label_f4[10:470, 10:470]
#         img_f4 = cv2.resize(img_f4, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f4 = cv2.resize(label_f4, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         augmented_imgs.append(img_f4)
#         augmented_labels.append(label_f4)
#
#         img_f5 = cv2.flip(img, 0).astype('uint8')
#         label_f5 = cv2.flip(label, 0).astype('uint8')
#         img_f5 = img_f5[10:470, 10:470]
#         label_f5 = label_f5[10:470, 10:470]
#         img_f5 = cv2.resize(img_f5, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f5 = cv2.resize(label_f5, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         augmented_imgs.append(img_f5)
#         augmented_labels.append(label_f5)
#
#         img_f6 = cv2.flip(img, 0).astype('uint8')
#         label_f6 = cv2.flip(label, 0).astype('uint8')
#         img_f6 = img_f6[10:470, :]
#         label_f6 = label_f6[10:470, :]
#         img_f6 = cv2.resize(img_f6, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f6 = cv2.resize(label_f6, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         augmented_imgs.append(img_f6)
#         augmented_labels.append(label_f6)
#
#         img_f7 = cv2.flip(img, 0).astype('uint8')
#         label_f7 = cv2.flip(label, 0).astype('uint8')
#         img_f7 = img_f7[:, 10:470]
#         label_f7 = label_f7[:, 10:470]
#         img_f7 = cv2.resize(img_f7, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f7 = cv2.resize(label_f7, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         augmented_imgs.append(img_f7)
#         augmented_labels.append(label_f7)
#         label_count = cv2.countNonZero(label)
#         if 5 < label_count < 1000:
#             try:
#                 cnts, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#                 xxxyyy = np.asarray(cnts[0])
#                 a_sque = np.squeeze(xxxyyy)
#                 rect = cv2.minAreaRect(a_sque)
#                 box = cv2.boxPoints(rect)
#                 box = np.intp(box)
#                 h1,h2 = min(box[0][1],box[1][1],box[2][1],box[3][1]),max(box[0][1],box[1][1],box[2][1],box[3][1])
#                 w1,w2 = min(box[0][0],box[1][0],box[2][0],box[3][0]),max(box[0][0],box[1][0],box[2][0],box[3][0])
#                 center_h,center_w = int((h1+h2)/2),int((w1+w2)/2)
#                 randn1 = random.randint(150, 160)
#                 randn1_1 = random.randint(150, 160)
#                 size_h = min(randn1,center_h,480-center_h)
#                 size_w = min(randn1_1, center_w, 480 - center_w)
#                 image2 = img[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
#                 label2 = label[(center_h-size_h):(center_h+size_h ), (center_w-size_w):(center_w+size_w)]
#                 image2 = cv2.resize(image2,(480,480),interpolation = cv2.INTER_LINEAR).astype('uint8')
#                 label2 = cv2.resize(label2, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
#                 augmented_imgs.append(image2)
#                 augmented_labels.append(label2)
#             except:
#                 pass
#
#         return augmented_imgs, augmented_labels

    # def __getitem__(self, index):
    #     img = self.imgs[index]
    #     img = img / 1.0  # 这行可能不需要，根据需要删除
    #     img = np.expand_dims(img, 2)
    #     img = np.tile(img, 3)
    #     label = (self.labels[index]).astype('uint8')
    #     label[label > 0] = 1
    #
    #     # 添加条件，只有在满足条件时才进行数据增强
    #     label_count = cv2.countNonZero(label)
    #     if 5 < label_count < 1000:
    #         # 调用数据增强方法
    #         augmented_imgs, augmented_labels = self.augment_data(img, label)
    #     else:
    #         # 如果不满足条件，保持原始图像和标签不变
    #         augmented_imgs = [img]
    #         augmented_labels = [label]
    #
    #     augmented_data = []
    #     for img, label in zip(augmented_imgs, augmented_labels):
    #         data = self.transforms(image=img, mask=label)
    #         # augmented_data.append((data['image'], (data['mask']).long()))
    #     return data['image'], (data['mask']).long()
    #     # return augmented_data
    #
    # def __len__(self):
    #     return len(self.imgs)


class Mydataset_test(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):

        img_path = self.imgs[index]
        img_name = img_path.split('/')[-1]
        label_path = self.labels[index]
        # shutil.copy(img_path,'/mnt/ai2020/orton/codes/CCM-SEG/result/img/'+img_name)
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path,cv2.IMREAD_UNCHANGED).astype('uint8')
        label[label > 0] = 1
        img = img/1.0
        img = np.expand_dims(img,2)
        img = np.tile(img, 3)
        data = self.transforms(image=img,mask=label)
        # max_label = torch.max(data['mask'])
        return img_name,data['image'],(data['mask']).long()

    def __len__(self):
        return len(self.imgs)


class Mydataset_infer(data.Dataset):
    def __init__(self, img_paths,  transform):
        self.imgs = img_paths
        self.transforms = transform

    def __getitem__(self, index):
        img_path_here = self.imgs[index]
        img = np.load(img_path_here)#[:,:,::-1]
        data = self.transforms(image=img)

        return img_path_here,data['image']

    def __len__(self):
        return len(self.imgs)







def for_train_transform():
    # aug_size=int(size/10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
            max_pixel_value=1023.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform


test_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)

import os
def get_image_paths(dataroot):

    paths = []
    if dataroot is not None:
        paths_img = os.listdir(dataroot)
        for _ in sorted(paths_img):
            path = os.path.join(dataroot, _)
            paths.append(path)
    return paths
class Mydataset_for_pre(data.Dataset):
    def __init__(self, img_paths,  resize,transform = test_transform):
        self.imgs = get_image_paths(img_paths)
        self.transforms = transform
        self.resize = resize
    def __getitem__(self, index):
        img_path = self.imgs[index]

        img = cv2.resize(cv2.imread(img_path), (self.resize,self.resize))[:,:,::-1]
        img = self.transforms(image=img)

        return img['image'],#, label

    def __len__(self):
        return len(self.imgs)