import os
from tqdm import tqdm
import cv2

path = '//mnt/ai2020/orton/dataset/CCM/2D/val/image/'
result = '/mnt/ai2020/orton/dataset/CCM/2D/val/image_max1023/'

names = os.listdir(path)
for name in tqdm(names):
    label = cv2.imread(path+name,cv2.IMREAD_UNCHANGED)
    label[label>1023]=1023
    cv2.imwrite(result+name,label)