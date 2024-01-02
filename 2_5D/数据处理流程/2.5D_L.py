import os
from PIL import Image

# 遍历文件夹
folder_path = "/mnt/ai2022/zlx/dataset/CCM/2D/val/11"  # 替换为文件夹的路径
output_path = "/mnt/ai2022/zlx/dataset/CCM/2D/val/1"  # 替换为保存结果的路径

grouped_images = {}
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)
            image_name = file[:4]
            image_group = grouped_images.get(image_name, [])
            image_group.append(image_path)
            grouped_images[image_name] = image_group

for image_name, image_group in grouped_images.items():
    image_group.sort(key=lambda x: int(x[-6:-4]))  # 按图片命名最后两位从小到大排序

    for index, image_path in enumerate(image_group):
        image = Image.open(image_path).convert("I")

        if index == 0 or index == len(image_group) - 1:
            new_image = Image.new("I", image.size)
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    new_pixel = pixel + pixel
                    new_image.putpixel((x, y), min(new_pixel, 255))

            new_image_path = os.path.join(output_path, os.path.basename(image_path))
            new_image.save(new_image_path)
        else:
            previous_image_path = image_group[index - 1]
            next_image_path = image_group[index + 1]
            previous_image = Image.open(previous_image_path).convert("I")
            next_image = Image.open(next_image_path).convert("I")

            new_image = Image.new("I", image.size)
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    previous_pixel = previous_image.getpixel((x, y))
                    next_pixel = next_image.getpixel((x, y))
                    new_pixel = pixel + previous_pixel + next_pixel
                    new_image.putpixel((x, y), min(new_pixel, 255))

            new_image_path = os.path.join(output_path, os.path.basename(image_path))
            new_image.save(new_image_path)
