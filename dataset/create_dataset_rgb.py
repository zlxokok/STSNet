import torch
from torch.utils import data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import shutil
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
        # img = img/1.0
        # img = np.expand_dims(img,2)
        # img = np.tile(img, 3)
        label = (self.labels[index]).astype('uint8')
        label[label>0]=1
        # label[label==127] = 1
        # label[label==255] = 2
        data = self.transforms(image=img,mask=label)
        # labelll = torch.max(data['mask'])z
        return data['image'],(data['mask']).long()

    def __len__(self):
        return len(self.imgs)

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
        label[label>0]=1
        # label[label==127] = 1
        # label[label==255] = 2
        # img = img/1.0
        # img = np.expand_dims(img,2)
        # img = np.tile(img, 3)
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
        # A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
            max_pixel_value=255.0,
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