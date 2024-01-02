import os
import pandas as pd
import random
# path_img = './for_hua_lu_/train_image/'
# path_img =  '//mnt/ai2020/orton/dataset/garstric_new_delete_bad//val_mask/'
path_img ='//media/orton/DATADRIVE1/dataset/gastric/image/'
img_dir = os.listdir(path_img)
numbers_data = len(img_dir)
# print(img_dir)
test_for_csv = []
for i in range(len(img_dir)):
    test_for_csv.append(img_dir[i])
random.shuffle(test_for_csv)
random.shuffle(test_for_csv)
random.shuffle(test_for_csv)
# random.shuffle(test_for_csv)
df = pd.DataFrame()
df['image_name'] = test_for_csv[:int(0.8*numbers_data)]
# df['label'] = labels
df.to_csv('/mnt/Annotation/orton_ds/codes/my_second_semi_segmentation/tools_seg/gastric/train_3.csv',index=False)

df2 = pd.DataFrame()
df2['image_name'] = test_for_csv[int(0.8*numbers_data):]
# df['label'] = labels
df2.to_csv('/mnt/Annotation/orton_ds/codes/my_second_semi_segmentation/tools_seg/gastric/val_3.csv',index=False)
# results_file = open('/media/orton/DATADRIVE1/project_folder/for_test2/for_csv/fusai_remain.csv', 'w', newline='')
# csv_writer = csv.writer(results_file, dialect='excel')
# for row in test_for_csv:
#     csv_writer.writerow(row)