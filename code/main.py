import torch
import torch.nn as nn

import sklearn.metrics as metrics

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_img_list, get_model, get_data
from pylab import *
import random
from test import fast_test
plt.switch_backend('agg')

# --------------------------------------------------------------------------------

models_list = ['AG_Net']

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=150,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--data_path', type=str, default='../data/DRIVE',
                    help='dir of the all img')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=2,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='test8',
                    help='some description define your train')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='2',
                    help='the gpu used')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

model_name = models_list[args.model_id]
model = get_model(model_name)

model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if args.use_gpu:
    model.cuda()
    print('GPUs used: (%s)' % args.gpu_avaiable)
    print('------- success use GPU --------')

EPS = 1e-12
# define path
data_path = args.data_path
img_list = get_img_list(args.data_path, flag='train')
test_img_list = get_img_list(args.data_path, flag='test')


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()

IOU_best = 0

print ('This model is %s_%s_%s_%s' % (model_name, args.n_class, args.img_size,args.my_description))
if not os.path.exists(r'../models/%s_%s' % (model_name, args.my_description)):
    os.mkdir(r'../models/%s_%s' % (model_name, args.my_description))

with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'w+') as f:
    f.write('This model is %s_%s: ' % (model_name, args.my_description)+'\n')
    f.write('args: '+str(args)+'\n')
    f.write('train lens: '+str(len(img_list))+' | test lens: '+str(len(test_img_list)))
    f.write('\n\n---------------------------------------------\n\n')


for epoch in range(args.epochs):
    model.train()

    begin_time = time.time()
    print ('This model is %s_%s_%s_%s' % (
        model_name, args.n_class, args.img_size, args.my_description))
    random.shuffle(img_list)

    if 'arg' in args.data_path:
        if (epoch % 10 ==  0) and epoch != 0 and epoch < 400:
            args.lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i, (start, end) in enumerate(zip(range(0, len(img_list), args.batch_size),
                                         range(args.batch_size, len(img_list) + args.batch_size,
                                               args.batch_size))):
        path = img_list[start:end]
        img, gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)

        optimizer.zero_grad()

        out, side_5, side_6, side_7, side_8 = model(img)
        out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, gt)
        loss += criterion(torch.log(softmax_2d(side_5) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_6) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_7) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_8) + EPS), gt)
        out = torch.log(softmax_2d(side_8) + EPS)

        loss.backward()
        optimizer.step()

        ppi = np.argmax(out.cpu().data.numpy(), 1)

        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        meanIU, Acc,Se,Sp,IU = calculate_Accuracy(my_confusion)

        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
                  '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name, args.my_description,epoch, i, loss.data[0], Acc,Se,Sp,
                                                                                  IU[0], IU[1]))

    print('training finish, time: %.1f s' % (time.time() - begin_time))

    if epoch % 10 == 0 and epoch != 0:
        torch.save(model.state_dict(),
                   '../models/%s_%s/%s.pth' % (model_name, args.my_description,str(epoch)))
        print('success save Nucleus_best model')
































