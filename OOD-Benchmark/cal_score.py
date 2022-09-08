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
    args.cur_temperature=args.temperature
    in_dataloader=get_dataloader(args,'in')
    get_realData_score(args,'{}_out'.format(args.in_dataset),in_dataloader,len(in_dataloader.dataset))
    out_dataset_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN']
    if args.in_dataset=='cifar10':
        out_dataset_list.append('cifar100')
    elif args.in_dataset=='cifar100':
        out_dataset_list.append('cifar10')
    for item in out_dataset_list:
        args.out_dataset=item
        if item == "SVHN":
            out_dataset = torchvision.datasets.SVHN(root='../data', split='train', download=True,transform=TrainTransform_list[args.in_dataset])
            out_dataloader = torch.utils.data.DataLoader(out_dataset, batch_size=args.ood_batch_size, shuffle=False)
        elif item=='cifar10':
            out_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=transforms_list[args.in_dataset])
            out_dataloader = torch.utils.data.DataLoader(out_dataset,batch_size=args.ood_batch_size, shuffle=False)
        elif item=='cifar100':
            out_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=transforms_list[args.in_dataset])
            out_dataloader = torch.utils.data.DataLoader(out_dataset,batch_size=args.ood_batch_size, shuffle=False)
        else:
            out_dataset = torchvision.datasets.ImageFolder("../data/{}".format(item), transform=transforms_list[args.in_dataset])
            out_dataloader = torch.utils.data.DataLoader(out_dataset, batch_size=args.ood_batch_size, shuffle=False)
        get_realData_score(args,item,out_dataloader,len(out_dataset))

    in_dataloader=get_dataloader(args,'in')
    args.cur_temperature=1
    get_realData_score(args,'{}_noise'.format(args.in_dataset),in_dataloader,len(in_dataloader.dataset))

    noise_dataset_list=['Gaussian', 'Uniform']
    args.cur_temperature=1
    for item in noise_dataset_list:
        args.out_dataset=item
        get_noiseData_score(args,item,10000)

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
def get_top_acc(args):
    in_dataloader=get_dataloader(args,'in')
    nTotal = len(in_dataloader.dataset)
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    net=load_net(args,model_path)
    with torch.no_grad():
        if args.infer=='bn':
            net.eval()
        correct_top1,correct_top5 = 0,0
        for data, target in in_dataloader:
            data, target = data.cuda(), target.cuda()
            outputs = net(data)[2]
            softmax_output=F.softmax(outputs,dim=1)
            pred_top1 = softmax_output.data.max(1)[1]
            correct_top1 += pred_top1.eq(target.data).cpu().sum().item()
            pred_top5 = softmax_output.topk(5,dim=1)[1]
            target_top5=torch.unsqueeze(target,dim=1).expand_as(pred_top5)
            correct_top5+=pred_top5.eq(target_top5).cpu().sum().item()
            # print(correct_top1,correct_top5)

    acc_top1,acc_top5 = round(100.*correct_top1/nTotal,2),round(100.*correct_top5/nTotal,2)
    print(acc_top1,acc_top5)
    with open('{}/top_acc.txt'.format(args.save),'wt') as f:
        print('accs: {}'.format(args.save),file=f)
        print('acc_top1:{},  acc_top5:{}'.format(acc_top1,acc_top5),file=f)
    
    return acc_top1,acc_top5

def get_realData_score(args,ds,dataloader,N):
    t0 = time.time()
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    if not os.path.exists('{}/score'.format(args.save)):
        os.makedirs('{}/score'.format(args.save))

    net=load_net(args,model_path)

    print("{}\nProcessing: {}, magnitude:{:.4f}, temperature:{}".format(args.save,ds,args.magnitude,args.cur_temperature))
    Pred,Pred_Tsc=torch.empty(0,dtype=torch.float64).cuda(),torch.empty(0,dtype=torch.float64).cuda()
    with torch.no_grad():
        if args.infer=='bn':
            net.eval()
        for batch_idx, data in enumerate(dataloader):
            if batch_idx>=int(10000/args.ood_batch_size): break
            images, _ = data
            inputs = images.cuda()
            outputs = net(inputs)[2]
            softmax_output=F.softmax(outputs,dim=1).to(float64)
            Pred=torch.cat((Pred,softmax_output),dim=0)
        
            # Using temperature scaling
            outputs_t = outputs / args.cur_temperature
            softmax_output_t=F.softmax(outputs_t,dim=1).to(float64)
            Pred_Tsc=torch.cat((Pred_Tsc,softmax_output_t),dim=0)

            if (batch_idx+1) % 10 == 0:
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(args.ood_batch_size*(batch_idx)+len(inputs), N,time.time()-t0))
                t0 = time.time()
    torch.save(Pred,"{}/score/{}_Pred.pt".format(args.save,ds))
    torch.save(Pred_Tsc,"{}/score/{}_Pred_Tsc.pt".format(args.save,ds))





def get_noiseData_score(args,ds,N):
    t0 = time.time()
    model_path="{}/latest-{}.pth".format(args.save,args.epochs)
    if not os.path.exists('{}/score'.format(args.save)):
        os.makedirs('{}/score'.format(args.save))

    net=load_net(args,model_path)
    print("{}\nProcessing: {}, magnitude:{:.4f}, temperature:{}".format(args.save,ds,args.magnitude,args.cur_temperature))

    Pred,Pred_Tsc=torch.empty(0,dtype=float64).cuda(),torch.empty(0,dtype=float64).cuda()
    with torch.no_grad():
        if args.infer=='bn':
            net.eval()
        for batch_idx in range (int(N/args.ood_batch_size)):
            if args.out_dataset=='Gaussian':
                images = torch.randn(100,3,32,32) + 0.5
                images = torch.clamp(images, 0, 1)
            elif args.out_dataset=='Uniform':
                images = torch.rand(100,3,32,32)
            images[:,0] = (images[:,0] - 125.3/255) / (63.0/255)
            images[:,1] = (images[:,1] - 123.0/255) / (62.1/255)
            images[:,2] = (images[:,2] - 113.9/255) / (66.7/255)
            
            inputs = images.cuda()
            outputs = net(inputs)[2]
            softmax_output=F.softmax(outputs,dim=1).to(float64)
            Pred=torch.cat((Pred,softmax_output),dim=0)
        
            # Using temperature scaling
            outputs_t = outputs / args.cur_temperature
            softmax_output_t=F.softmax(outputs_t,dim=1).to(float64)
            Pred_Tsc=torch.cat((Pred_Tsc,softmax_output_t),dim=0)

            if (batch_idx+1) % 10 == 0:
                print("{:4}/{:4} images processed, {:.1f} seconds used.".format(args.ood_batch_size*(batch_idx)+len(inputs), N,time.time()-t0))
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
















