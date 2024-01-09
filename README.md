STSNet
The codes for the work "STSNet
:

1. Prepare data
You can go to https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset to acquire the BUID dataset.

2. Train/Test
Run the train script on the ISIC-2017 and the COVID-QU-Ex dataset. The batch size we used is 8. If you do not have enough GPU memory, the bacth size can be reduced to 4 or 6 to save memory. For more information, contact 1154692412@qq.com.

Train

python train.py 
Test
python test.py 
References
TransUnet
SwinTransformer
Swin-Unet
