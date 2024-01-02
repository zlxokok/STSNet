import os
import cv2
import numpy as np
import nrrd

# 设置输入和输出路径
input_folder = '/mnt/ai2022/zlx/dataset/CCM/25D/yuce_2D/DeepLabv3_mobilenetv2_2.5D_CE'
output_folder = '/mnt/ai2022/zlx/dataset/CCM/25D/yuce_3Dquan/DeepLabv3_mobilenetv2_2.5D_CE'

# 获取文件夹下所有图片文件的路径
image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.png')]
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 按命名前四位分组
image_groups = {}
for file in image_files:
    filename = os.path.basename(file)
    group_name = filename[:4]
    if group_name not in image_groups:
        image_groups[group_name] = []
    image_groups[group_name].append(file)

# 对每个组的图片进行排序
for group_name, files in image_groups.items():
    files.sort(key=lambda x: int(x[-6:-4]))  # 按命名最后两位从小到大排序
    i = len(files)
    # 创建空白的(640, 640, 22)图像
    result_img = np.zeros((480, 480, i), dtype=np.uint16)

    # 逐个组合图片
    for i, file in enumerate(reversed(files)):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
        result_img[:, :, i] = img


    # 保存为NRRD文件
    output_filename = group_name + '_img_seg'+'.nrrd'
    output_path = os.path.join(output_folder, output_filename)
    nrrd.write(output_path, result_img)

    print(f"Group {group_name}: Saved as {output_filename}")