STSNet

The codes for the work "STSNet
:

1. Prepare data
You can go to https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset to acquire the BUID dataset.
cd /2.5D and Run the 3_augu_labeled2.py andg 4_find_label_center_together.py to preprocess the 2D data
3. Train/Test
Run the train script on the BUID dataset. The batch size we used is 4. If you do not have enough GPU memory, the bacth size can be reduced to 6 or 8 to save memory. For more information, contact 1154692412@qq.com.

Train

python main_DeepLabv3.py 
Test
python infer.py 

