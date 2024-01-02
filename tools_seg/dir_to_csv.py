import os
import random

import pandas as pd

path ='//mnt/ai2020/orton/dataset/CCM/2D_labeled_augu/train/image_together/'
names = os.listdir(path)
random.shuffle(names)
df = pd.DataFrame()
df['image_name'] = names
# df['label'] = label_train
df.to_csv('/mnt/ai2020/orton/codes/CCM-SEG/dataset/train_all.csv',index=False)
print('fds')