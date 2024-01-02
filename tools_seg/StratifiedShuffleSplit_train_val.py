import pandas as pd
import numpy
# import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
train_imgs_list = '/mnt/ai2020/orton/codes/3D_my_second_semi_segmentation/dateset/BraTS2019/data_csv/all.csv'
data =  pd.read_csv(train_imgs_list)
names = data['image_name']
# aa = data[0]
kfold= KFold(n_splits=2,shuffle=True,random_state=0)  #     StratifiedShuffleSplit
i = 0
for train_index,test_index in kfold.split(names):

    name_train,name_val = names[train_index].tolist(),names[test_index].tolist()
    # print('fds')
    df = pd.DataFrame()
    df['image_name'] = name_train
    df.to_csv('/mnt/ai2020/orton/codes/3D_my_second_semi_segmentation/dateset/BraTS2019/data_csv/divide'+str(i)+'train_half.csv',index=False)

    df2 = pd.DataFrame()
    df2['image_name'] = name_val
    df2.to_csv('//mnt/ai2020/orton/codes/3D_my_second_semi_segmentation/dateset/BraTS2019/data_csv/divide'+str(i)+'val_half.csv',index=False)
    i +=1
    del df2,df
    # print(train_index,test_index)
print('fds')