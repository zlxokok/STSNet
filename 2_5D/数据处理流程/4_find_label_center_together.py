import random

from tqdm import tqdm
import nrrd
import os
import cv2
import numpy as np
import shutil


path_img = '/mnt/ai2022/zlx/dataset/xirou/TrainDataset/images/'
path_label = '/mnt/ai2022/zlx/dataset/xirou/TrainDataset/masks/'

result_img = '/mnt/ai2022/zlx/dataset/xirou/RGB/TrainDataset/images（复件）/'
result_label = '/mnt/ai2022/zlx/dataset/xirou/RGB/TrainDataset/masks（复件）/'
names = os.listdir(path_img)#[:3]

for name in tqdm(names):
    # shutil.copy(path_img+name,result_img+name)
    # shutil.copy(path_label + name, result_label + name)
    img = cv2.imread(path_img+name,cv2.IMREAD_UNCHANGED)
    label = cv2.imread(path_label+name,cv2.IMREAD_UNCHANGED)
    # label = cv2.imread('/mnt/ai2020/orton/dataset/CCM/2D_labeled_augu/label2/A021_img_3.png',cv2.IMREAD_UNCHANGED)
    # label[label > 0] = 255
    kernel = np.ones((2, 2), np.float32)
    label = cv2.dilate(label, kernel)
    # max_label = np.max(label)
    # if max_label > 0:
    label_count = cv2.countNonZero(label)
    if label_count > 5:
        try:
            cnts, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            xxxyyy = np.asarray(cnts[0])
            a_sque = np.squeeze(xxxyyy)
            # print(name,a_sque)
            # aa = len(a_sque)
            rect = cv2.minAreaRect(a_sque)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            h1,h2 = min(box[0][1],box[1][1],box[2][1],box[3][1]),max(box[0][1],box[1][1],box[2][1],box[3][1])
            w1,w2 = min(box[0][0],box[1][0],box[2][0],box[3][0]),max(box[0][0],box[1][0],box[2][0],box[3][0])
            center_h,center_w = int((h1+h2)/2),int((w1+w2)/2)

            randn1 = random.randint(150, 160)
            randn1_1 = random.randint(150, 160)
            size_h = min(randn1,center_h,480-center_h)
            size_w = min(randn1_1, center_w, 480 - center_w)
            image2 = img[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            label2 = label[(center_h-size_h):(center_h+size_h ), (center_w-size_w):(center_w+size_w)]
            image2 = cv2.resize(image2,(480,480),interpolation = cv2.INTER_LINEAR).astype('uint8')
            label2 = cv2.resize(label2, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
            cv2.imwrite(result_label+name.replace('.png','_cut1.png'),label2)
            cv2.imwrite(result_img + name.replace('.png', '_cut1.png'), image2)



            # randn2 = random.randint(60, 75)
            # randn2_1 = random.randint(60, 75)
            # size_h = min(randn2, center_h, 480 - center_h)
            # size_w = min(randn2_1, center_w, 480 - center_w)
            # image2 = img[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # label2 = label[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # image2 = cv2.resize(image2, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint16')
            # label2 = cv2.resize(label2, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
            # cv2.imwrite(result_label + name.replace('.png', '_cut2.png'), label2)
            # cv2.imwrite(result_img + name.replace('.png', '_cut2.png'), image2)
            #
            # randn3 = random.randint(75, 90)
            # randn3_1 = random.randint(75, 90)
            # size_h = min(randn3, center_h, 480 - center_h)
            # size_w = min(randn3_1, center_w, 480 - center_w)
            # image2 = img[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # label2 = label[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # image2 = cv2.resize(image2, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint16')
            # label2 = cv2.resize(label2, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
            # cv2.imwrite(result_label + name.replace('.png', '_cut3.png'), label2)
            # cv2.imwrite(result_img + name.replace('.png', '_cut3.png'), image2)
            #
            # randn3 = random.randint(50, 100)
            # randn3_1 = random.randint(50, 100)
            # size_h = min(randn3, center_h, 480 - center_h)
            # size_w = min(randn3_1, center_w, 480 - center_w)
            # image2 = img[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # label2 = label[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # image2 = cv2.resize(image2, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint16')
            # label2 = cv2.resize(label2, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
            # cv2.imwrite(result_label + name.replace('.png', '_cut4.png'), label2)
            # cv2.imwrite(result_img + name.replace('.png', '_cut4.png'), image2)
            #
            # randn3 = random.randint(60, 100)
            # randn3_1 = random.randint(60, 100)
            # size_h = min(randn3, center_h, 480 - center_h)
            # size_w = min(randn3_1, center_w, 480 - center_w)
            # image2 = img[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # label2 = label[(center_h - size_h):(center_h + size_h), (center_w - size_w):(center_w + size_w)]
            # image2 = cv2.resize(image2, (480, 480), interpolation=cv2.INTER_LINEAR).astype('uint16')
            # label2 = cv2.resize(label2, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
            # cv2.imwrite(result_label + name.replace('.png', '_cut5.png'), label2)
            # cv2.imwrite(result_img + name.replace('.png', '_cut5.png'), image2)
        except TypeError:
            print(name)
        # print('fds')