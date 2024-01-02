from tqdm import tqdm
import nrrd
import os
import cv2
import numpy as np
import shutil

path_img = '/mnt/ai2022/zlx/dataset/BUSI/image_512/'
path_label = '/mnt/ai2022/zlx/dataset/BUSI/mask_512/'

result_img = '/mnt/ai2022/zlx/dataset/BUSI/image_512（复件）/'
result_label = '/mnt/ai2022/zlx/dataset/BUSI/mask_512（复件）/'
names = os.listdir(path_img)  # [:3]
if not os.path.exists(result_img):
    os.mkdir(result_img)

if not os.path.exists(result_label):
    os.mkdir(result_label)


for name in tqdm(names):
    shutil.copy(path_img + name, result_img + name)
    shutil.copy(path_label + name, result_label + name)
    img = cv2.imread(path_img+name,cv2.IMREAD_UNCHANGED)
    label = cv2.imread(path_label+name,cv2.IMREAD_UNCHANGED)
    # label_count = cv2.countNonZero(label)
    label_count = cv2.countNonZero(label)

    if 2000 > label_count > 5:
        img = cv2.imread(path_img + name, cv2.IMREAD_UNCHANGED)
        img_f1 = cv2.flip(img, 0).astype('uint8')
        label_f1 = cv2.flip(label, 0).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu1.png'), img_f1)
        cv2.imwrite(result_label + name.replace('.png', '_augu1.png'), label_f1)

        img_f2 = cv2.flip(img, 1).astype('uint8')
        label_f2 = cv2.flip(label, 1).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu2.png'), img_f2)
        cv2.imwrite(result_label + name.replace('.png', '_augu2.png'), label_f2)

        img_f3 = img[100:500, 10:500]
        label_f3 = label[100:500, 10:500]
        img_f3 = cv2.resize(img_f3, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
        label_f3 = cv2.resize(label_f3, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu3.png'), img_f3)
        cv2.imwrite(result_label + name.replace('.png', '_augu3.png'), label_f3)

        img_f4 = cv2.flip(img, 1).astype('uint16')
        label_f4 = cv2.flip(label, 1).astype('uint8')
        img_f4 = img_f4[100:500, 10:500]
        label_f4 = label_f4[100:500, 10:500]
        img_f4 = cv2.resize(img_f4, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
        label_f4 = cv2.resize(label_f4, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
        cv2.imwrite(result_img + name.replace('.png', '_augu4.png'), img_f4)
        cv2.imwrite(result_label + name.replace('.png', '_augu4.png'), label_f4)

        # shutil.copy(path_img + name, result_img + name)
        # shutil.copy(path_label + name, result_label + name)
        # img = cv2.imread(path_img + name, cv2.IMREAD_UNCHANGED)
        # label = cv2.imread(path_label + name, cv2.IMREAD_UNCHANGED)
        # # label_count = cv2.countNonZero(label)
        # label_count = cv2.countNonZero(label)
        # # img = img.astype('uint8')
        # label = label.astype('uint8')
        # # cv2.imwrite(result_img + name.replace('.png', '.png'), img)
        # cv2.imwrite(result_label + name.replace('.png', '.png'), label)










        # img_f5 = cv2.flip(img, 0).astype('uint8')
        # label_f5 = cv2.flip(label, 0).astype('uint8')
        # img_f5 = img_f5[10:500, 10:500]
        # label_f5 = label_f5[10:500, 10:500]
        # img_f5 = cv2.resize(img_f5, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
        # label_f5 = cv2.resize(label_f5, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
        # cv2.imwrite(result_img + name.replace('.png', '_augu5.png'), img_f5)
        # cv2.imwrite(result_label + name.replace('.png', '_augu5.png'), label_f5)
        #
        # img_f6 = cv2.flip(img, 0).astype('uint8')
        # label_f6 = cv2.flip(label, 0).astype('uint8')
        # img_f6 = img_f6[10:500, :]
        # label_f6 = label_f6[10:500, :]
        # img_f6 = cv2.resize(img_f6, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
        # label_f6 = cv2.resize(label_f6, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
        # cv2.imwrite(result_img + name.replace('.png', '_augu6.png'), img_f6)
        # cv2.imwrite(result_label + name.replace('.png', '_augu6.png'), label_f6)
        #
        # img_f7 = cv2.flip(img, 0).astype('uint8')
        # label_f7 = cv2.flip(label, 0).astype('uint8')
        # img_f7 = img_f7[:, 10:500]
        # label_f7 = label_f7[:, 10:500]
        # img_f7 = cv2.resize(img_f7, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
        # label_f7 = cv2.resize(label_f7, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
        # cv2.imwrite(result_img + name.replace('.png', '_augu7.png'), img_f7)
        # cv2.imwrite(result_label + name.replace('.png', '_augu7.png'), label_f7)

        # print('fsd')
# from tqdm import tqdm
# import nrrd
# import os
# import cv2
# import numpy as np
# import shutil
#
# path_img = '/mnt/ai2022/zlx/dataset/Breast Ultrasound Images Dataset/processed/image/'
# path_label = '/mnt/ai2022/zlx/dataset/Breast Ultrasound Images Dataset/processed/mask/'
#
# result_img = '/mnt/ai2022/zlx/dataset/Breast Ultrasound Images Dataset/processededed/image_qie_1500/'
# result_label = '/mnt/ai2022/zlx/dataset/Breast Ultrasound Images Dataset/processededed/label_qie_1500/'
# names = os.listdir(path_img)  # [:3]
# if not os.path.exists(result_img):
#     os.mkdir(result_img)
#
# if not os.path.exists(result_label):
#     os.mkdir(result_label)
#
#
# for name in tqdm(names):
#     shutil.copy(path_img + name, result_img + name)
#     shutil.copy(path_label + name, result_label + name)
#     # img = cv2.imread(path_img+name,cv2.IMREAD_UNCHANGED)
#     label = cv2.imread(path_label+name,cv2.IMREAD_UNCHANGED)
#     # label_count = cv2.countNonZero(label)
#     label_count = cv2.countNonZero(label)
#
#     if 1500 > label_count > 5:
#         img = cv2.imread(path_img + name, cv2.IMREAD_UNCHANGED)
#         # img_f1 = cv2.flip(img, 0).astype('uint8')
#         # label_f1 = cv2.flip(label, 0).astype('uint8')
#         # cv2.imwrite(result_img + name.replace('.png', '_augu1.png'), img_f1)
#         # cv2.imwrite(result_label + name.replace('.png', '_augu1.png'), label_f1)
#         #
#         # img_f2 = cv2.flip(img, 1).astype('uint8')
#         # label_f2 = cv2.flip(label, 1).astype('uint8')
#         # cv2.imwrite(result_img + name.replace('.png', '_augu2.png'), img_f2)
#         # cv2.imwrite(result_label + name.replace('.png', '_augu2.png'), label_f2)
#
#         img_f3 = img[30:482, 30:482]
#         label_f3 = label[30:482, 30:482]
#         img_f3 = cv2.resize(img_f3, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f3 = cv2.resize(label_f3, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         cv2.imwrite(result_img + name.replace('.png', '_augu3.png'), img_f3)
#         cv2.imwrite(result_label + name.replace('.png', '_augu3.png'), label_f3)
#
#         # img_f4 = cv2.flip(img, 1).astype('uint8')
#         # label_f4 = cv2.flip(label, 1).astype('uint8')
#         img_f4 = img[40:472, 40:472]
#         label_f4 = label[40:472, 40:472]
#         img_f4 = cv2.resize(img_f4, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f4 = cv2.resize(label_f4, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         cv2.imwrite(result_img + name.replace('.png', '_augu4.png'), img_f4)
#         cv2.imwrite(result_label + name.replace('.png', '_augu4.png'), label_f4)
#
#         # img_f5 = cv2.flip(img, 0).astype('uint8')
#         # label_f5 = cv2.flip(label, 0).astype('uint8')
#         img_f5 = img[50:462, 50:462]
#         label_f5 = label[50:462, 50:462]
#         img_f5 = cv2.resize(img_f5, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f5 = cv2.resize(label_f5, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         cv2.imwrite(result_img + name.replace('.png', '_augu5.png'), img_f5)
#         cv2.imwrite(result_label + name.replace('.png', '_augu5.png'), label_f5)
#
#         # img_f6 = cv2.flip(img, 0).astype('uint8')
#         # label_f6 = cv2.flip(label, 0).astype('uint8')
#         img_f6 = img[60:452, 60:452]
#         label_f6 = label[60:452, 60:452]
#         img_f6 = cv2.resize(img_f6, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f6 = cv2.resize(label_f6, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         cv2.imwrite(result_img + name.replace('.png', '_augu6.png'), img_f6)
#         cv2.imwrite(result_label + name.replace('.png', '_augu6.png'), label_f6)
#
#         # img_f7 = cv2.flip(img, 0).astype('uint8')
#         # label_f7 = cv2.flip(label, 0).astype('uint8')
#         img_f7 = img[70:442, 70:442]
#         label_f7 = label[70:442, 70:442]
#         img_f7 = cv2.resize(img_f7, (512, 512), interpolation=cv2.INTER_LINEAR).astype('uint8')
#         label_f7 = cv2.resize(label_f7, (512, 512), interpolation=cv2.INTER_NEAREST).astype('uint8')
#         cv2.imwrite(result_img + name.replace('.png', '_augu7.png'), img_f7)
#         cv2.imwrite(result_label + name.replace('.png', '_augu7.png'), label_f7)
#
#         # print('fsd')
