from __future__ import division
import numpy as np
import os
import cv2
import torch
from models import *
import cPickle as pkl
from torch.autograd import Variable


def get_data(data_path, img_name, img_size=256, gpu=True,flag=False):

    def get_label(label):
        tmp_gt = label.copy()
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    labels = []
    tmp_gts = []

    img_shape =[]
    label_ori = []

    batch_size = len(img_name)
    for i in range(batch_size):
        img_path = os.path.join(data_path, 'images', img_name[i])
        label_path = os.path.join(data_path, 'label', img_name[i][:-3]+'png')

        img = cv2.imread(img_path)
        label = cv2.imread(label_path,0)

        img_shape.append(img.shape)
        label_ori.append(label)
        label[label == 255] = 1

        img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        if gpu:
            img = img.cuda()

        label, tmp_gt = get_label(label)
        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)

    images = torch.stack(images)
    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    if flag:
        label_ori = np.stack(label_ori)

    return images, labels, tmp_gts, img_shape,label_ori


def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)

    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])

    return  meanIU,Acc,Se,Sp,IU


def get_model(model_name):
    if model_name=='AG_Net':
        return AG_Net


def get_img_list(data_path, flag='train'):
    with open(os.path.join(data_path, 'train_dict.pkl')) as f:
        train_dict = pkl.load(f)
    if flag=='train':
        img_list = train_dict['train_list']
    elif flag=='test':
        img_list = train_dict['test_list']
    elif flag=='val':
        img_list = train_dict['val_list']
    else:
        img_list = train_dict['test_list']+train_dict['train_list']
    return img_list
