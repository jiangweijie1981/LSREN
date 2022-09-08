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
    eval_info=torch.load('{}/eval_info.pt'.format(args.save))
    ood_info=torch.load('{}/ood_info.pt'.format(args.save))
    args.magnitude=eval_info['magnitude']

    in_dataloader=get_dataloader(args,'in')
    get_realData_score(args,args.in_dataset,in_dataloader)

    val_in_dataloader=get_dataloader(args,'val_in')
    get_realData_score(args,'{}_val_in'.format(args.in_dataset),val_in_dataloader)

    out_dataloader=get_dataloader(args,'out')
    get_realData_score(args,'{}_out'.format(args.in_dataset),out_dataloader) 

    val_dataloader=get_dataloader(args,'val')
    get_realData_score(args,'{}_val'.format(args.in_dataset),val_dataloader) 

    mean_std=torch.load('{}/mean_std.pt'.format(args.save))
    te_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std['mean'],mean_std['std'])
        ])
    out_datasets=['Imagenet','Imagenet_resize','LSUN','LSUN_resize','iSUN']
    for item in out_datasets:
        out_dataset = torchvision.datasets.ImageFolder("../data/{}".format(item), transform=te_transform)
        out_dataloader = torch.utils.data.DataLoader(out_dataset, batch_size=args.ood_batch_size, shuffle=False)
        get_realData_score(args,'{}_{}'.format(args.in_dataset,item),out_dataloader) 
   
    if 'eval' in args.inferance:
        eval_info['acc']=get_eval_acc(args)
        torch.save(eval_info,'{}/eval_info.pt'.format(args.save))
    if 'ood' in args.inferance:
        ood_info['acc']=get_ood_acc(args)
        torch.save(ood_info,'{}/ood_info.pt'.format(args.save))

def get_eval_acc(args):
    in_dataloader=get_dataloader(args,'in')
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    net=load_net(args,model_path)
    with torch.no_grad():
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
def get_ood_acc(args):
    in_dataloader=get_dataloader(args,'in')
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    net=load_net(args,model_path)
    with torch.no_grad():
        # net.eval()
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
    if 'eval' in args.inferance:
        eval_net=load_net(args,model_path)
        if not os.path.exists('{}/eval_score'.format(args.save)):
            os.makedirs('{}/eval_score'.format(args.save))
        print("{}\nProcessing: {}, magnitude:{:.4f}, temperature:{}".format(args.save,ds,args.magnitude,args.temperature))
        Pred,Pred_Tsc=torch.empty(0,dtype=float64).cuda(),torch.empty(0,dtype=float64).cuda()
        with torch.no_grad():
            eval_net.eval()
            for batch_idx, data in enumerate(dataloader):
                # if batch_idx>=int(10000/args.ood_batch_size): break
                images, _ = data
                inputs = images.cuda()
                outputs = eval_net(inputs)[2]
                softmax_output=F.softmax(outputs,dim=1).to(float64)
                Pred=torch.cat((Pred,softmax_output),dim=0)
                outputs_t = outputs / args.temperature
                softmax_output_t=F.softmax(outputs_t,dim=1).to(float64)
                Pred_Tsc=torch.cat((Pred_Tsc,softmax_output_t),dim=0)
                if (batch_idx+1) % 10 == 0:
                    print("{}/{} images processed, {:.1f} seconds used.".format(batch_idx+1, len(dataloader),time.time()-t0))
                    t0 = time.time()
        torch.save(Pred,"{}/eval_score/{}_Pred.pt".format(args.save,ds))
        torch.save(Pred_Tsc,"{}/eval_score/{}_Pred_Tsc.pt".format(args.save,ds))

        
    if 'ood' in args.inferance:
        ood_net=load_net(args,model_path)
        if not os.path.exists('{}/ood_score'.format(args.save)):
            os.makedirs('{}/ood_score'.format(args.save))
        print("{}\nProcessing: {}, magnitude:{:.4f}, temperature:{}".format(args.save,ds,args.magnitude,args.temperature))
        Pred,Pred_Tsc=torch.empty(0,dtype=float64).cuda(),torch.empty(0,dtype=float64).cuda()
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                # if batch_idx>=int(10000/args.ood_batch_size): break
                images, _ = data
                inputs = images.cuda()
                outputs = ood_net(inputs)[2]
                softmax_output=F.softmax(outputs,dim=1).to(float64)
                Pred=torch.cat((Pred,softmax_output),dim=0)
                outputs_t = outputs / args.temperature
                softmax_output_t=F.softmax(outputs_t,dim=1).to(float64)
                Pred_Tsc=torch.cat((Pred_Tsc,softmax_output_t),dim=0)
                if (batch_idx+1) % 10 == 0:
                    print("{}/{} images processed, {:.1f} seconds used.".format(batch_idx+1, len(dataloader),time.time()-t0))
                    t0 = time.time()
        torch.save(Pred,"{}/ood_score/{}_Pred.pt".format(args.save,ds))
        torch.save(Pred_Tsc,"{}/ood_score/{}_Pred_Tsc.pt".format(args.save,ds))



















