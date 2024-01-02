from tqdm import tqdm
import nrrd
import os
import cv2
import numpy as np
import shutil

path_img = '/mnt/ai2022/zlx/dataset/CCM补充/T2/25D/train/image/'
path_label = '/mnt/ai2022/zlx/dataset/CCM补充/T2/25D/train/label/'

result_img = '/mnt/ai2022/zlx/dataset/CCM补充/T2/25D/train/image_1000_2/'
result_label = '/mnt/ai2022/zlx/dataset/CCM补充/T2/25D/train/label_1000_2/'
names = os.listdir(path_img)#[:3]
if not os.path.exists(result_img):
    os.mkdir(result_img)

if not os.path.exists(result_label):
    os.mkdir(result_label)



for name in tqdm(names):
    shutil.copy(path_img+name,result_img+name)
    shutil.copy(path_label + name, result_label + name)
    # img = cv2.imread(path_img+name,cv2.IMREAD_UNCHANGED)
    label = cv2.imread(path_label+name,cv2.IMREAD_UNCHANGED)
    label_count = cv2.countNonZero(label)
    if 1000 > label_count > 5:
        img = cv2.imread(path_img + name,cv2.IMREAD_UNCHANGED)
        img_f1 = cv2.flip(img, 0).astype('uint16')
        label_f1 = cv2.flip(label, 0).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png','_augu1.png'),img_f1)
        cv2.imwrite(result_label + name.replace('.png', '_augu1.png'), label_f1)

        img_f2 = cv2.flip(img, 1).astype('uint16')
        label_f2 = cv2.flip(label, 1).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu2.png'), img_f2)
        cv2.imwrite(result_label + name.replace('.png', '_augu2.png'), label_f2)

        img_f3 = img[50:430,50:430]
        label_f3 = label[50:430,50:430]
        img_f3 = cv2.resize(img_f3, (480, 480)).astype('uint16')
        label_f3 = cv2.resize(label_f3, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu3.png'), img_f3)
        cv2.imwrite(result_label + name.replace('.png', '_augu3.png'), label_f3)

        img_f4 = cv2.flip(img, 1).astype('uint16')
        label_f4 = cv2.flip(label, 1).astype('uint8')
        img_f4 = img_f4[50:430,50:430]
        label_f4 = label_f4[50:430,50:430]
        img_f4 = cv2.resize(img_f4, (480, 480)).astype('uint16')
        label_f4 = cv2.resize(label_f4, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu4.png'), img_f4)
        cv2.imwrite(result_label + name.replace('.png', '_augu4.png'), label_f4)

        img_f5 = cv2.flip(img, 0).astype('uint16')
        label_f5 = cv2.flip(label, 0).astype('uint8')
        img_f5 = img_f5[50:430,50:430]
        label_f5 = label_f5[50:430,50:430]
        img_f5 = cv2.resize(img_f5, (480, 480)).astype('uint16')
        label_f5 = cv2.resize(label_f5, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu5.png'), img_f5)
        cv2.imwrite(result_label + name.replace('.png', '_augu5.png'), label_f5)

        img_f6 = cv2.flip(img, 0).astype('uint16')
        label_f6 = cv2.flip(label, 0).astype('uint8')
        img_f6 = img_f6[50:430, :]
        label_f6 = label_f6[50:430, :]
        img_f6 = cv2.resize(img_f6, (480, 480)).astype('uint16')
        label_f6 = cv2.resize(label_f6, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu6.png'), img_f6)
        cv2.imwrite(result_label + name.replace('.png', '_augu6.png'), label_f6)

        img_f7 = cv2.flip(img, 0).astype('uint16')
        label_f7 = cv2.flip(label, 0).astype('uint8')
        img_f7 = img_f7[:, 50:430]
        label_f7 = label_f7[:, 50:430]
        img_f7 = cv2.resize(img_f7, (480, 480)).astype('uint16')
        label_f7 = cv2.resize(label_f7, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu7.png'), img_f7)
        cv2.imwrite(result_label + name.replace('.png', '_augu7.png'), label_f7)


        img_f3 = img[75:405, 75:405]
        label_f3 = label[75:405, 75:405]
        img_f3 = cv2.resize(img_f3, (480, 480)).astype('uint16')
        label_f3 = cv2.resize(label_f3, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu8.png'), img_f3)
        cv2.imwrite(result_label + name.replace('.png', '_augu8.png'), label_f3)

        img_f4 = cv2.flip(img, 1).astype('uint16')
        label_f4 = cv2.flip(label, 1).astype('uint8')
        img_f4 = img_f4[75:405, 75:405]
        label_f4 = label_f4[75:405, 75:405]
        img_f4 = cv2.resize(img_f4, (480, 480)).astype('uint16')
        label_f4 = cv2.resize(label_f4, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu9.png'), img_f4)
        cv2.imwrite(result_label + name.replace('.png', '_augu9.png'), label_f4)

        img_f3 = img[100:380, 100:380]
        label_f3 = label[100:380, 100:380]
        img_f3 = cv2.resize(img_f3, (480, 480)).astype('uint16')
        label_f3 = cv2.resize(label_f3, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu10.png'), img_f3)
        cv2.imwrite(result_label + name.replace('.png', '_augu10.png'), label_f3)

        img_f4 = cv2.flip(img, 1).astype('uint16')
        label_f4 = cv2.flip(label, 1).astype('uint8')
        img_f4 = img_f4[100:380, 100:380]
        label_f4 = label_f4[100:380, 100:380]
        img_f4 = cv2.resize(img_f4, (480, 480)).astype('uint16')
        label_f4 = cv2.resize(label_f4, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu11.png'), img_f4)
        cv2.imwrite(result_label + name.replace('.png', '_augu11.png'), label_f4)

        img_f3 = img[125:355, 125:355]
        label_f3 = label[125:355, 125:355]
        img_f3 = cv2.resize(img_f3, (480, 480)).astype('uint16')
        label_f3 = cv2.resize(label_f3, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu12.png'), img_f3)
        cv2.imwrite(result_label + name.replace('.png', '_augu12.png'), label_f3)

        img_f4 = cv2.flip(img, 1).astype('uint16')
        label_f4 = cv2.flip(label, 1).astype('uint8')
        img_f4 = img_f4[125:355, 125:355]
        label_f4 = label_f4[125:355, 125:355]
        img_f4 = cv2.resize(img_f4, (480, 480)).astype('uint16')
        label_f4 = cv2.resize(label_f4, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu13.png'), img_f4)
        cv2.imwrite(result_label + name.replace('.png', '_augu13.png'), label_f4)

        img_f3 = img[25:455, 25:455]
        label_f3 = label[25:455, 25:455]
        img_f3 = cv2.resize(img_f3, (480, 480)).astype('uint16')
        label_f3 = cv2.resize(label_f3, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu14.png'), img_f3)
        cv2.imwrite(result_label + name.replace('.png', '_augu14.png'), label_f3)

        img_f4 = cv2.flip(img, 1).astype('uint16')
        label_f4 = cv2.flip(label, 1).astype('uint8')
        img_f4 = img_f4[25:455, 25:455]
        label_f4 = label_f4[25:455, 25:455]
        img_f4 = cv2.resize(img_f4, (480, 480)).astype('uint16')
        label_f4 = cv2.resize(label_f4, (480, 480), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu15.png'), img_f4)
        cv2.imwrite(result_label + name.replace('.png', '_augu15.png'), label_f4)
        # print('fsd')