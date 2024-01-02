import random
import numpy as np
import torch,os
# import matplotlib.pyplot as plt
def set_seed(seed=1): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def write_options(model_savedir,args,test_acc):
    aaa = []
    aaa.append(['lr',str(args.lr)])
    aaa.append(['batch',args.batch_size])
    aaa.append(['save_name',args.save_name])
    aaa.append(['seed',args.batch_size])
    aaa.append(['best_iou',str(test_acc)])
    aaa.append(['warm_epoch',args.warm_epoch])
    aaa.append(['end_epoch',args.end_epoch])
    f = open(model_savedir+'option'+'.txt', "a")
    for option_things in aaa:
        f.write(str(option_things)+'\n')
    f.close()

def write_options_with_test(model_savedir,args,test_acc,average_test_iou,average_test_Pre,average_test_recall,
                            average_test_F1score,average_test_pa):
    aaa = []
    aaa.append(['lr',str(args.lr)])
    aaa.append(['resize',str(args.resize)])
    aaa.append(['batch',args.batch_size])
    aaa.append(['save_name',args.save_name])
    aaa.append(['seed',args.batch_size])
    aaa.append(['best_iou',str(test_acc)])
    aaa.append(['warm_epoch',args.warm_epoch])
    aaa.append(['end_epoch',args.end_epoch])
    aaa.append(['average_test_iou',average_test_iou])
    aaa.append(['average_test_Pre',average_test_Pre])
    aaa.append(['average_test_recall',average_test_recall])
    aaa.append(['average_test_F1score',average_test_F1score])
    aaa.append(['average_test_pa',average_test_pa])
    f = open(model_savedir+'option'+'.txt', "a")
    for option_things in aaa:
        f.write(str(option_things)+'\n')
    f.close()

# def output_figure(train_loss,val_loss,train_acc,val_acc,model_savedir):
#     fig = plt.figure(figsize=(22,8))
#     plt.plot(train_loss, label='train loss')
#     plt.plot(val_loss, label='valid loss')
#     # plt.plot(test_loss, label='test loss')
#     plt.legend()
#     plt.title('Loss Curve')
#     plt.savefig(''.join([model_savedir, '_Loss.png']), bbox_inches = 'tight')
#
#     fig = plt.figure(figsize=(22,8))
#     plt.plot(train_acc, label='train acc')
#     plt.plot(val_acc, label='valid acc')
#     # plt.plot(test_acc, label='test acc')
#     plt.legend()
#     plt.title('Accuracy Curve')
#     plt.savefig(''.join([model_savedir, '_Acc.png']), bbox_inches = 'tight')
