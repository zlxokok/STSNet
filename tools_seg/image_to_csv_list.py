import os
import glob
import pandas as pd
import random
# path_img = './for_hua_lu_/train_image/'
# path_img =  '//mnt/ai2020/orton/dataset/garstric_new_delete_bad//val_mask/'
path_img ='//mnt/ai2020/orton/codes/3D_my_second_semi_segmentation/dateset/BraTS2019/data/'
# img_dir = os.listdir(path_img)
img_dir = glob.glob(path_img+'*')
print(img_dir)
test_for_csv = []
for i in range(len(img_dir)):
    test_for_csv.append(img_dir[i])
# random.shuffle(test_for_csv)
# random.shuffle(test_for_csv)
# random.shuffle(test_for_csv)
# random.shuffle(test_for_csv)
df = pd.DataFrame()
df['image_name'] = test_for_csv
# df['label'] = labels
df.to_csv('/mnt/ai2020/orton/codes/3D_my_second_semi_segmentation/dateset/BraTS2019/data_csv/all.csv',index=False)

# results_file = open('/media/orton/DATADRIVE1/project_folder/for_test2/for_csv/fusai_remain.csv', 'w', newline='')
# csv_writer = csv.writer(results_file, dialect='excel')
# for row in test_for_csv:
#     csv_writer.writerow(row)