import os
import random
import shutil

path_img =  '/mnt/ai2022/zlx/dataset/kuang/dataset-gt/Image/'
path_label = '/mnt/ai2022/zlx/dataset/kuang/dataset-gt/GT/'

result_img =  '/mnt/ai2022/zlx/dataset/kuang/test/image/'
result_label = '/mnt/ai2022/zlx/dataset/kuang/test/mask/'

# result_img2 =  '/mnt/ai2022/zlx/dataset/TNUS/test/image/'
# result_label2 = '/mnt/ai2022/zlx/dataset/TNUS/test/label/'

names = os.listdir(path_img)
random.shuffle(names)
lens = len(names)
rate = int(0.1*lens)
# rate2 = int(0.2*lens)
names_move = names[:rate]
# names_move2 = names[lens - rate2:]


for name in names_move:
    shutil.move(path_img+name,result_img+name)
    shutil.move(path_label + name.replace('.png','.png'), result_label + name.replace('.png','.png'))


# for name in names_move2:
#     shutil.move(path_img+name,result_img2+name)
#     shutil.move(path_label + name, result_label2 + name)




