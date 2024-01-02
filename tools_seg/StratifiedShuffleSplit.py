import pandas as pd
import numpy
import numpy as np
# import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
train_imgs_list = '//mnt/ai2021/orton/second/3D_my_second_semi_segmentation/dateset/BraTS2019/data_base/train.csv'
data =  pd.read_csv(train_imgs_list)
names = data['image_name']

label_list = []
for name in names:
    label_list.append(name[8])
# aa = data[0]
lelabel = LabelEncoder()
labels = lelabel.fit_transform(label_list)
# size_train = 0.7
kfold= ShuffleSplit(n_splits=5,train_size=0.9,test_size=0.1,random_state=0)
# kfold= KFold(n_splits=5,shuffle=True,random_state=0) #     StratifiedShuffleSplit
i = 0
for train_index,test_index in kfold.split(names):
    # print('train_index',train_index,'test_index',test_index)
    names = np.array(names)
    name_train,name_val = names[train_index].tolist(),names[test_index].tolist()
    label_train,label_val = labels[train_index].tolist(),labels[test_index].tolist()
    # print('fds')
    df = pd.DataFrame()
    df['image_name'] = name_train
    df['label'] = label_train
    df.to_csv('//mnt/ai2021/orton/second/3D_my_second_semi_segmentation/dateset/fold'+str(i)+'/fold'+str(i)+'train09.csv',index=False)
    #
    # df2 = pd.DataFrame()
    # df2['image_name'] = name_val
    # df2['label'] = label_val
    # df2.to_csv('//mnt/ai2021/orton/second/3D_my_second_semi_segmentation/dateset/fold'+str(i)+'/fold'+str(i)+'train01_remain09.csv',index=False)
    i +=1
    # print(train_index,test_index)
print('fds')