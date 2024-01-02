import nrrd
import os
import cv2
import numpy as np

path_img = '/mnt/ai2022/zlx/dataset/Cross-Modality Domain Adaptation for Medical Image Segmentation_miccai_2022/crossmoda2022_training/training_target/'
# path_label = '/mnt/ai2022/zlx/dataset/CCM补充/T2/8:2/3d/val/label/'
result_path_img = '/mnt/ai2022/zlx/dataset/Cross-Modality Domain Adaptation for Medical Image Segmentation_miccai_2022/crossmoda2022_training/T2/'
# result_path_label = '/mnt/ai2022/zlx/dataset/CCM补充/T2/8:2/val/label/'
names = os.listdir(path_img)  # [:3]
for name in names:
    img = nrrd.read(path_img + name)[0]
    # mask = nrrd.read(path_label + name.replace('.nrrd', '_seg.nrrd'))[0]
    size = img.shape[2]
    # img2 = img[200:400,200:400,10]
    for i in range(size):
        if i == 0:
            patch_img = img[:, :, :2]
        elif i == (size - 1):
            patch_img = img[:, :, -2:]
        else:
            patch_img = img[:, :, i]
            patch_img = cv2.resize(patch_img, (640, 640), interpolation=cv2.INTER_NEAREST).astype('uint16')
            max_h = np.max(patch_img)
            if max_h > 10000:
                print(name)
            patch_img = patch_img[80:560, 80:560]
            # patch_label = mask[:, :, i]
            patch_label = cv2.resize(patch_label, (640, 640), interpolation=cv2.INTER_NEAREST) * 120
            patch_label = patch_label[80:560, 80:560]

            cv2.imwrite(result_path_img+str(name.replace('.nrrd','_'))+str(i)+'.png',patch_img)
            # cv2.imwrite(result_path_label+str(name.replace('.nrrd','_'))+str(i)+'.png',patch_label)

# aa = np.max(mask)
# print(aa)
print('fsd')
