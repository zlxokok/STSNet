import os, random, shutil
# def moveFile(fileimgDir,filemaskDir,file_hue_Dir,tarimgDir,tarmaskDir,tar_hue_Dir):
def moveFile(fileimgDir,filemaskDir,tarimgDir,tarmaskDir):
# def moveFile(fileimgDir,dir,tarimgDir):
    pathDir = os.listdir(fileimgDir)    #取图片的原始路径
    filenumber=len(pathDir)
    rate=0.1 #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    print (sample)
    i = 0
    for name in sample:
        # a = name[0]
        # # print(a)
        # if int(a) > 0:
            name_mask = name.replace('.jpg','.png')
            # name_new  = str(i)+'.png
            shutil.move(fileimgDir+name, tarimgDir+name)
            shutil.move(filemaskDir+name_mask, tarmaskDir+name_mask)
            i +=1
            # shutil.copy(filemaskDir+name, tarmaskDir+name)
        # shutil.move(file_hue_Dir+name, tar_hue_Dir+name)
    return

if __name__ == '__main__':
    fileimgDir = "//mnt/ai2020/orton/dataset/garstric/orton/orton/train_image/"     #源图片文件夹路径
    filemaskDir = "//mnt/ai2020/orton/dataset/garstric/orton/orton/train_mask/"
    # file_hue_Dir = "//media/orton/DATADRIVE1/project_folder/for_test2/for_hua_lu_/resized_big3/train/image_hue/"
    # dirpre = '/media/orton/DATADRIVE1/hua_lu_bei/result_0805/result915/result/result/'
    tarimgDir = "//mnt/ai2020/orton/dataset/garstric/orton/orton/test_image/"   #移动到新的文件夹路径
    tarmaskDir ='/mnt/ai2020/orton/dataset/garstric/orton/orton/test_mask/'
    # tar_hue_Dir = "//media/orton/DATADRIVE1/project_folder/for_test2/for_hua_lu_/resized_big3/eval/image_hue/"
    moveFile(fileimgDir,filemaskDir,tarimgDir,tarmaskDir)
    # moveFile(tarimgDir,tarmaskDir,fileimgDir,filemaskDir)
    # moveFile(fileimgDir,filemaskDir,file_hue_Dir,tarimgDir,tarmaskDir,tar_hue_Dir)
    # moveFile(fileimgDir,dirpre,tarimgDir)