import torch
import numpy as np
import SimpleITK as sitk

def mean_dice(pred, target, num_classes=3):
    dice_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)
        dice_score = (2.0 * intersection) / (union + 1e-7)  # Add a small epsilon to avoid division by zero
        dice_scores.append(dice_score)
    mean_dice = np.mean(dice_scores)
    return mean_dice

def mean_iou(pred, target, num_classes=3):
    iou_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask) - intersection
        iou_score = intersection / (union + 1e-7)  # Add a small epsilon to avoid division by zero
        iou_scores.append(iou_score)
    mean_iou = np.mean(iou_scores)
    return mean_iou

# def compute_mean_dice(labels_true, labels_pred, num_classes):
#     mean_dice = 0.0
#
#     for class_id in range(num_classes):
#         if labels_true == 2:
#             print("good")
#         intersection = np.sum((labels_true == class_id) & (labels_pred == class_id))
#         dice = (2.0 * intersection) / (np.sum(labels_true == class_id) + np.sum(labels_pred == class_id))
#         mean_dice += dice
#
#     mean_dice /= num_classes
#     return mean_dice
#
# def compute_mean_iou(labels_true, labels_pred, num_classes):
#     mean_iou = 0.0
#
#     for class_id in range(num_classes):
#         intersection = np.sum((labels_true == class_id) & (labels_pred == class_id))
#         union = np.sum((labels_true == class_id) | (labels_pred == class_id))
#         iou = intersection / union
#         mean_iou += iou
#
#     mean_iou /= num_classes
#     return mean_iou

def calculate_miou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''

    inputTmp = torch.zeros([input.shape[0],classNum, input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    # target = target.unsqueeze(1)#同上
    input = input.to(torch.int64)
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    target = target.to(torch.int64)
    target = target.unsqueeze(1)  # 同上
    target = target.to('cuda')
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)


def calculate_mdice(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = 2 * torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)


def Pa(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    tmp = input == target
    x=torch.sum(tmp).float()
    y=input.nelement()
    # print('x',x,y)
    return (x / y)
def pre(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP+1e-6)/(TP+FP+1e-6)
    return pre
def recall(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    recall=(TP+1e-6)/(TP+FN+ 1e-6)
    return recall
def F1score(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP+1e-6) / (TP + FP + 1e-6)
    recall=(TP+1e-6)/(TP+FN+ 1e-6)
    F1score=(2*(pre)*(recall))/(pre+recall+1e-6)
    return F1score

def jaccard(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    ja = TP/((TP+FN+FP)+1e-5)
    return ja

def accuracy(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    AC = (TP+TN)/((TP+FP+TN+FN+1e-5))
    return AC

def dice(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    DI = 2*TP/((2*TP+FN+FP+1e-5))
    return DI

def recall(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    SE = TP/(TP+FN+1e-5)
    return SE

def SP(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    SP = TN/((TN+FP)+1e-5)
    return SP

def precision(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    precision = TP/(TP+FP+1e-5)
    return precision






def calculate_fwiou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)#同上
    batchFwious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        fwious = []
        for j in range(classNum):#遍历类别，包括背景
            TP_FN = torch.sum(targetOht[i][j])
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            fwiou = (TP_FN/(input.shape[2]*input.shape[3])) * iou
            fwious.append(fwiou.item())
        fwiou = np.mean(fwious)#计算该图像的miou
        # print(miou)
        batchFwious.append(fwiou)
    return np.mean(batchFwious)

def calculate_dice(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection) / (union + 1e-8)
    return dice


def calculate_jaccard(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    jaccard = intersection / (union + 1e-8)
    return jaccard


def calculate_asd(pred, target, spacing):
    pred_sitk = sitk.GetImageFromArray(pred)
    target_sitk = sitk.GetImageFromArray(target)
    pred_sitk.SetSpacing(spacing)
    target_sitk.SetSpacing(spacing)
    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_image_filter.Execute(pred_sitk > 0, target_sitk > 0)
    asd = hausdorff_distance_image_filter.GetAverageSurfaceDistance()
    return asd


def calculate_hd(pred, target, spacing):
    pred_sitk = sitk.GetImageFromArray(pred)
    target_sitk = sitk.GetImageFromArray(target)
    pred_sitk.SetSpacing(spacing)
    target_sitk.SetSpacing(spacing)
    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_image_filter.Execute(pred_sitk > 0, target_sitk > 0)
    hd = hausdorff_distance_image_filter.GetHausdorffDistance()
    return hd