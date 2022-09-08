# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""
# import torch
from torch import float64
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from configs import *
from get_data import *
from utils import *

def cal_score(args):
    info=torch.load('{}/info.pt'.format(args.save))
    args.magnitude=info['magnitude']
    in_dataloader=get_dataloader(args,'in')
    get_realData_score(args,args.in_dataset,in_dataloader)
    if args.in_dataset=='cifar4':
        out_dataloader=get_dataloader(args,'out10')
        get_realData_score(args,'{}_out10'.format(args.in_dataset),out_dataloader)
        out_dataloader=get_dataloader(args,'out50')
        get_realData_score(args,'{}_out50'.format(args.in_dataset),out_dataloader)
    else:
        out_dataloader=get_dataloader(args,'out')
        get_realData_score(args,'{}_out'.format(args.in_dataset),out_dataloader) 
    out_dataloader=get_dataloader(args,'val')
    get_realData_score(args,'{}_val'.format(args.in_dataset),out_dataloader)  
    info['acc']=get_acc(args)
    torch.save(info,'{}/info.pt'.format(args.save))

def get_acc(args):
    in_dataloader=get_dataloader(args,'in')
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    net=load_net(args,model_path)
    with torch.no_grad():
        if args.infer=='bn':
            net.eval()
        incorrect = 0
        for data, target in in_dataloader:
            data, target = data.cuda(), target.cuda()
            outputs = net(data)[2]
            softmax_output=F.softmax(outputs,dim=1)
            pred = softmax_output.data.max(1)[1]
            incorrect += pred.ne(target.data).cpu().sum()
    nTotal = len(in_dataloader.dataset)
    err = 100.*incorrect/nTotal
    return 100-err

def get_realData_score(args,ds,dataloader):
    t0 = time.time()
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    if not os.path.exists('{}/score'.format(args.save)):
        os.makedirs('{}/score'.format(args.save))
    print("{}\nProcessing: {}, magnitude:{:.4f}, temperature:{}".format(args.save,ds,args.magnitude,args.temperature))

    criterion = nn.CrossEntropyLoss()
    net=load_net(args,model_path)

    Pred,Pred_Tsc=torch.empty(0,dtype=float64).cuda(),torch.empty(0,dtype=float64).cuda()
    with torch.no_grad():
        if args.infer=='bn':
            net.eval()
        for batch_idx, data in enumerate(dataloader):
            # if batch_idx>=int(10000/args.ood_batch_size): break
            images, _ = data
            inputs = images.cuda()
            outputs = net(inputs)[2]
            softmax_output=F.softmax(outputs,dim=1).to(float64)
            Pred=torch.cat((Pred,softmax_output),dim=0)
            outputs_t = outputs / args.temperature
            softmax_output_t=F.softmax(outputs_t,dim=1).to(float64)
            Pred_Tsc=torch.cat((Pred_Tsc,softmax_output_t),dim=0)
            if (batch_idx+1) % 10 == 0:
                print("{}/{} images processed, {:.1f} seconds used.".format(batch_idx+1, len(dataloader),time.time()-t0))
                t0 = time.time()
    torch.save(Pred,"{}/score/{}_Pred.pt".format(args.save,ds))
    torch.save(Pred_Tsc,"{}/score/{}_Pred_Tsc.pt".format(args.save,ds))







if __name__ == '__main__':
    args=Config().data
    set_seed(args.seed)
    smooth_list=get_smooth_list(args)
    if args.train_type=='LSR':
        for i in range(len(smooth_list)):
            args.smooth=smooth_list[i]
            args.save = '../models/{}/{}_{}{}/{}_{}/seed{}_epochs{}_s{:.2f}'.format(
                args.net,args.in_dataset,args.noise_type,args.noise_ratio,
                args.train_type,args.ensemble_num,args.seed,args.epochs,args.smooth)
            cal_score(args)
    if args.train_type=='ILSR':
        for i in range(len(args.penalty_list)):
            args.penalty=args.penalty_list[i]
            for j in range(len(smooth_list)):
                args.smooth=smooth_list[j]
                args.save = '../models/{}/{}_{}{}/{}_{}_{}/seed{}_epochs{}_s{:.2f}'.format(
                    args.net,args.in_dataset,args.noise_type,args.noise_ratio,
                    args.train_type,args.ensemble_num,args.penalty,args.seed,args.epochs,args.smooth)
                cal_score(args)
















