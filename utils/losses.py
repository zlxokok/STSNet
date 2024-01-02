import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def dice_loss(score, target):
    # target = target.float()
    # smooth = 1e-5
    # intersect = torch.sum(score * target)
    # y_sum = torch.sum(target * target)
    # z_sum = torch.sum(score * score)
    # loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    # loss = 1 - loss
    # return loss
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)

    TP = intersect
    white_pixels = torch.sum(target)

    if white_pixels > 1000 and TP == 1:
        weight = 1
    elif 5 < white_pixels <= 1000 and TP == 1:
        weight = 1
    else:
        weight = 0
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    weighted_loss = loss * weight
    loss = 1 - weighted_loss

    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class CELoss(nn.Module):
    """Cross Entropy Loss with weighted classes
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
    Returns:
        The weighted cross entropy loss
    """

    def __init__(self, weight=None, ignore_index=None):
        super(CELoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                ce_loss = F.cross_entropy(predict[:, i], target[:, i], reduction='mean')

                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], got [{}]'.format(target.shape[1], self.weight.shape[0])

                    # Count the number of white label pixels in the original image
                    num_white_pixels = torch.sum(target[:, i] == 1)

                    # Calculate true positives (TP)
                    TP = torch.sum((predict[:, i].argmax(dim=1) == 1) & (target[:, i] == 1))

                    # Update weight based on conditions
                    if num_white_pixels > 1000 and TP == 1:
                        ce_loss *= 1
                    elif 5 < num_white_pixels < 1000 and TP == 1:
                        ce_loss *= 5

                total_loss += ce_loss

        return total_loss / target.shape[1]



class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        target = target.unsqueeze(1)
        target = target.repeat(1, 2, 1, 1)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])

                    # Count the number of white label pixels in the original image
                    num_white_pixels = torch.sum(target[:, 1] == 1)

                    # Calculate true positives (TP) by finding the intersection between predict and target
                    TP = torch.sum((predict[:, i] == 1) & (target[:, i] == 1))

                    # Update weight based on conditions
                    if num_white_pixels > 1000 and (TP / num_white_pixels) >= 0.8:
                        dice_loss *= 1
                    elif 5 < num_white_pixels < 1000 and (TP / num_white_pixels) >= 0.8:
                        dice_loss *= 5

                total_loss += dice_loss

        return total_loss / target.shape[1]


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map
