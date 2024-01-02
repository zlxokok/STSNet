import os

import pandas as pd

path ='/mnt/ai2021/orton/second/my_2_seg_covid/image_npy'
names = os.listdir(path)

df = pd.DataFrame()
df['image_name'] = names
# df['label'] = label_train
df.to_csv('//mnt/ai2021/orton/second/my_2_seg_covid/dataset/train_all.csv',index=False)
print('fds')