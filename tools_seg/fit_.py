import cv2
import os
import random
import torch
from tqdm import tqdm
from tools_seg.Miou import Pa,calculate_miou
from torch.cuda.amp import autocast, GradScaler
import argparse


def fit(epoch,epochs, model, trainloader, valloader,device,criterion,optimizer,CosineLR):
    # with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
    scaler = GradScaler()
    if torch.cuda.is_available():
        model.to('cuda')
    running_loss = 0
    model.train()
    train_pa_whole = 0
    train_iou_whole = 0
    for  batch_idx, (imgs, masks) in enumerate(trainloader):
        # t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
        imgs, masks_cuda = imgs.to(device), masks.to(device)
        imgs = imgs.float()
        with autocast():
            masks_pred = model(imgs)
            loss = criterion(masks_pred, masks_cuda)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # with torch.no_grad():
        predicted = masks_pred.argmax(1)
        train_pa = Pa(predicted,masks_cuda)
        train_iou = calculate_miou(predicted,masks_cuda,2)
        train_pa_whole += train_pa.item()
        train_iou_whole += train_iou.item()
        running_loss += loss.item()
        # epoch_acc = train_pa_whole/(batch_idx+1)
        epoch_iou = train_iou_whole/(batch_idx+1)
        # t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
        #               train_pa='{:.2f}%'.format(epoch_acc*100),train_iou='{:.2f}%'.format(epoch_iou*100))
        # t.update(1)
    # epoch_acc = correct / total
    epoch_loss = running_loss / len(trainloader.dataset)
# with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
    val_running_loss = 0
    val_pa_whole = 0
    val_iou_whole = 0
    model.eval()
    with torch.no_grad():
        for batch_idx,(imgs, masks) in enumerate(valloader):
            # t.set_description("val(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            masks_pred = model(imgs)
            predicted = masks_pred.argmax(1)
            val_pa = Pa(predicted,masks_cuda)
            val_iou = calculate_miou(predicted,masks_cuda,2)
            val_pa_whole += val_pa.item()
            val_iou_whole += val_iou.item()
            loss = criterion(masks_pred, masks_cuda)
            val_running_loss += loss.item()
            epoch_val_acc = val_pa_whole/(batch_idx+1)
            epoch_val_iou = val_iou_whole/(batch_idx+1)
            # t.set_postfix(loss='{:.3f}'.format(val_running_loss / (batch_idx + 1)),
            #               val_pa='{:.2f}%'.format(epoch_val_acc*100),val_iou='{:.2f}%'.format(epoch_val_iou*100))
            # t.update(1)
        # epoch_test_acc = test_correct / test_total
    epoch_val_loss = val_running_loss / len(valloader.dataset)
    #if epoch > 2:
    CosineLR.step()
        #if epoch > 2:

    return epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou

def fit_musheng(epoch,epochs, model, trainloader, valloader,device,criterion,optimizer,CosineLR):
    with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
        scaler = GradScaler()
        if torch.cuda.is_available():
            model.to('cuda')
        running_loss = 0
        model.train()
        train_pa_whole = 0
        train_iou_whole = 0
        for  batch_idx, (imgs, masks) in enumerate(trainloader):
            t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda = imgs.to(device), masks.to(device)
            imgs = imgs.float()
            with autocast():
                masks_pred = model(imgs)
                loss = criterion(masks_pred, masks_cuda)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # with torch.no_grad():
            predicted = masks_pred.argmax(1)
            train_pa = Pa(predicted,masks_cuda)
            train_iou = calculate_miou(predicted,masks_cuda,2)
            train_pa_whole += train_pa.item()
            train_iou_whole += train_iou.item()
            running_loss += loss.item()
            epoch_acc = train_pa_whole/(batch_idx+1)
            epoch_iou = train_iou_whole/(batch_idx+1)
            t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
                          train_pa='{:.2f}%'.format(epoch_acc*100),train_iou='{:.2f}%'.format(epoch_iou*100))
            t.update(1)
        # epoch_acc = correct / total
        epoch_loss = running_loss / len(trainloader.dataset)
    with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
        val_running_loss = 0
        val_pa_whole = 0
        val_iou_whole = 0
        model.eval()
        with torch.no_grad():
            for batch_idx,(imgs, masks) in enumerate(valloader):
                t.set_description("val(Epoch{}/{})".format(epoch, epochs))
                imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
                imgs = imgs.float()
                masks_pred = model(imgs)
                predicted = masks_pred.argmax(1)
                val_pa = Pa(predicted,masks_cuda)
                val_iou = calculate_miou(predicted,masks_cuda,2)
                val_pa_whole += val_pa.item()
                val_iou_whole += val_iou.item()
                loss = criterion(masks_pred, masks_cuda)
                val_running_loss += loss.item()
                epoch_val_acc = val_pa_whole/(batch_idx+1)
                epoch_val_iou = val_iou_whole/(batch_idx+1)
                t.set_postfix(loss='{:.3f}'.format(val_running_loss / (batch_idx + 1)),
                              val_pa='{:.2f}%'.format(epoch_val_acc*100),val_iou='{:.2f}%'.format(epoch_val_iou*100))
                t.update(1)
        # epoch_test_acc = test_correct / test_total
        epoch_val_loss = val_running_loss / len(valloader.dataset)
        #if epoch > 2:
        CosineLR.step()
        #if epoch > 2:

        return epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc



def fit_with_seg(epoch,epochs, model, trainloader, valloader,device,criterion,optimizer,CosineLR):
    with tqdm(total=len(trainloader), ncols=160, ascii=True) as t:
        scaler = GradScaler()
        if torch.cuda.is_available():
            model.to('cuda')
        running_loss,running_loss_seg,running_loss_class = 0,0,0
        model.train()
        train_pa_whole = 0
        train_iou_whole = 0
        correct ,total= 0,0
        for  batch_idx, (imgs, masks,class_label) in enumerate(trainloader):
            t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda,class_label_cuda = imgs.to(device), masks.to(device),class_label.to(device)
            imgs = imgs.float()
            with autocast():
                masks_pred,class_pred = model(imgs)
                loss_seg = criterion(masks_pred, masks_cuda)
                loss_class = criterion(class_pred, class_label_cuda.squeeze())
                loss = loss_seg * 0.8 + loss_class*0.2
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # with torch.no_grad():
            predicted = masks_pred.argmax(1)
            train_pa = Pa(predicted,masks_cuda)
            train_iou = calculate_miou(predicted,masks_cuda,2)
            train_pa_whole += train_pa.item()
            train_iou_whole += train_iou.item()
            running_loss += loss.item()
            running_loss_seg += loss_seg.item()
            running_loss_class += loss_class.item()

            epoch_pa = train_pa_whole/(batch_idx+1)
            epoch_iou = train_iou_whole/(batch_idx+1)

            with torch.no_grad():
                correct += (torch.max(class_pred, 1)[1].view(class_label_cuda.size()).data == class_label_cuda.data).sum()
                total += trainloader.batch_size

            epoch_acc = 100*((correct.item()) / total)

            t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
                          train_pa='{:.2f}%'.format(epoch_pa*100),train_iou='{:.2f}%'.format(epoch_iou*100),
                          class_acc='{:.2f}%'.format(epoch_acc))
            t.update(1)
        # epoch_acc = correct / total
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_loss_seg = running_loss_seg / len(trainloader.dataset)
        epoch_loss_class = running_loss_class / len(trainloader.dataset)
    with tqdm(total=len(valloader), ncols=160, ascii=True) as t:
        val_running_loss , val_running_loss_seg , val_running_loss_class= 0,0,0
        val_pa_whole = 0
        val_iou_whole = 0
        correct_val = 0
        # total_val_acc = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for batch_idx,(imgs, masks,class_label) in enumerate(valloader):
                t.set_description("val(Epoch{}/{})".format(epoch, epochs))
                imgs, masks_cuda,class_label_cuda = imgs.to(device), masks.to(device),class_label.to(device)
                imgs = imgs.float()
                masks_pred ,class_pred= model(imgs)
                predicted = masks_pred.argmax(1)
                val_pa = Pa(predicted,masks_cuda)
                val_iou = calculate_miou(predicted,masks_cuda,2)
                val_pa_whole += val_pa.item()
                val_iou_whole += val_iou.item()

                loss_seg = criterion(masks_pred, masks_cuda)
                loss_class = criterion(class_pred, class_label_cuda.squeeze())
                loss = loss_seg*0.8 + loss_class*0.2
                val_running_loss += loss.item()
                val_running_loss_seg += loss_seg.item()
                val_running_loss_class += loss_class.item()

                epoch_val_pa = val_pa_whole/(batch_idx+1)
                epoch_val_iou = val_iou_whole/(batch_idx+1)

                correct_val += (torch.max(class_pred, 1)[1].view(class_label_cuda.size()).data == class_label_cuda.data).sum()
                total_val += valloader.batch_size
                total_val_acc = 100*((correct_val.item()) / total_val)

                t.set_postfix(loss='{:.3f}'.format(val_running_loss / (batch_idx + 1)),
                              val_pa='{:.2f}%'.format(epoch_val_pa*100),val_iou='{:.2f}%'.format(epoch_val_iou*100),
                              val_acc='{:.2f}%'.format(total_val_acc))
                t.update(1)
        # epoch_test_acc = test_correct / test_total
        epoch_val_loss = val_running_loss / len(valloader.dataset)
        epoch_val_loss_seg = val_running_loss / len(valloader.dataset)
        epoch_val_loss_calss = val_running_loss / len(valloader.dataset)
        #if epoch > 2:
        CosineLR.step()
        #if epoch > 2:

        return epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou,   \
               epoch_loss_seg,epoch_loss_class,epoch_val_loss_seg,epoch_val_loss_calss,\
               epoch_acc,total_val_acc