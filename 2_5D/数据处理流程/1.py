import os
import cv2
import nibabel as nib
import numpy as np

path_img = '/mnt/ai2022/zlx/dataset/Cross-Modality Domain Adaptation for Medical Image Segmentation_miccai_2022/crossmoda2022_training/training_target/'
result_path_img = '/mnt/ai2022/zlx/dataset/Cross-Modality Domain Adaptation for Medical Image Segmentation_miccai_2022/crossmoda2022_training/T2/'
names = os.listdir(path_img)

for name in names:
    img_nii = nib.load(os.path.join(path_img, name))
    img = img_nii.get_fdata()
    size = img.shape[2]

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

            cv2.imwrite(result_path_img + str(name.replace('.nii', '_')) + str(i) + '.png', patch_img)

print('fsd')
