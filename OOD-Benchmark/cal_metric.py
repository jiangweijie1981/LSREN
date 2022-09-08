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

import torch
import os
import numpy as np
import time

def cal_metric(args,metric_type_list):
    if 'tnr95' in metric_type_list:
        tnr95(args)
    if 'auroc' in metric_type_list:
        auroc(args)
    if 'auprIn' in metric_type_list:
        auprIn(args)
    if 'auprOut' in metric_type_list:
        auprOut(args)
    if 'detec' in metric_type_list:
        detection(args)
    return

def tnr95(args):
    if args.save.find('ensemble')>0:
        info=torch.load('{}/info_{}.pt'.format(args.save,args.calibration))
    else:
        info=torch.load('{}/info.pt'.format(args.save))
    #test modify 5-4
    out_dataset_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN']
    # out_dataset_list=['Imagenet']
    if args.in_dataset=='cifar10':
        out_dataset_list.append('cifar100')
    elif args.in_dataset=='cifar100':
        out_dataset_list.append('cifar10')
    noise_dataset_list=['Gaussian', 'Uniform']
    
    if args.save.find('ensemble')>0:
        score_type_list=['cali_Pred','cali_Pred_Tsc']
        for score_type in score_type_list:
            in_score = torch.sort(torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
            for item in out_dataset_list:
                out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
                info['{}_tnr95_{}'.format(item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
            in_score = torch.sort(torch.load('{}/score/{}_noise_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
            for item in noise_dataset_list:
                out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
                info['{}_tnr95_{}'.format(item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
    score_type_list=['Pred','Pred_Tsc']
    for score_type in score_type_list:
        soft_max = torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
        in_score = torch.sort(torch.sum(soft_max*torch.log(soft_max),dim=1))[0]
        for item in out_dataset_list:
            soft_max = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            info['{}_tnr95_{}'.format(item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
        soft_max = torch.load('{}/score/{}_noise_{}.pt'.format(args.save,args.in_dataset,score_type))
        in_score = torch.sort(torch.sum(soft_max*torch.log(soft_max),dim=1))[0]
        for item in noise_dataset_list:
            soft_max = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            info['{}_tnr95_{}'.format(item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
    if args.save.find('ensemble')>0:
        torch.save(info,"{}/info_{}.pt".format(args.save,args.calibration))
    else:
        torch.save(info,"{}/info.pt".format(args.save))
    print(info)

def auroc(args):
    t0 = time.time()
    #test modify 5-5
    out_dataset_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN']
    # out_dataset_list=['Imagenet']
    if args.in_dataset=='cifar10':
        out_dataset_list.append('cifar100')
    elif args.in_dataset=='cifar100':
        out_dataset_list.append('cifar10')
    noise_dataset_list=['Gaussian', 'Uniform']
    if args.save.find('ensemble')>0:
        info=torch.load('{}/info_{}.pt'.format(args.save,args.calibration))
    else:
        info=torch.load('{}/info.pt'.format(args.save))
    
    if args.save.find('ensemble')>0:
        score_type_list=['cali_Pred','cali_Pred_Tsc']
        for score_type in score_type_list:
            in_score = torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
            for item in out_dataset_list:
                print('\ncal auroc:{}/score/{}_{}.pt'.format(args.save,item,score_type))
                out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
                thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                auroc_score = 0.0
                fprTemp = 1.0
                for thred_idx in range (len(thred)):
                    tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                    fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                    auroc_score += (-fpr+fprTemp)*tpr
                    fprTemp = fpr
                auroc_score += fpr * tpr
                info['{}_auroc_{}'.format(item,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))
            in_score = torch.load('{}/score/{}_noise_{}.pt'.format(args.save,args.in_dataset,score_type))
            for item in noise_dataset_list:
                print('\ncal auroc:{}/score/{}_{}.pt'.format(args.save,item,score_type))
                out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
                thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                auroc_score = 0.0
                fprTemp = 1.0
                for thred_idx in range (len(thred)):
                    tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                    fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                    auroc_score += (-fpr+fprTemp)*tpr
                    fprTemp = fpr
                auroc_score += fpr * tpr
                info['{}_auroc_{}'.format(item,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))

    score_type_list=['Pred','Pred_Tsc']
    for score_type in score_type_list:
        soft_max = torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
        in_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
        for item in out_dataset_list:
            print('\ncal auroc:{}/score/{}_{}.pt'.format(args.save,item,score_type))
            soft_max = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
            auroc_score = 0.0
            fprTemp = 1.0
            for thred_idx in range (len(thred)):
                tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                auroc_score += (-fpr+fprTemp)*tpr
                fprTemp = fpr
            auroc_score += fpr * tpr
            info['{}_auroc_{}'.format(item,score_type)] = auroc_score
            print('use time:{:.1f} second'.format(time.time()-t0))
        soft_max = torch.load('{}/score/{}_noise_{}.pt'.format(args.save,args.in_dataset,score_type))
        in_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
        for item in noise_dataset_list:
            print('\ncal auroc:{}/score/{}_{}.pt'.format(args.save,item,score_type))
            soft_max = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
            auroc_score = 0.0
            fprTemp = 1.0
            for thred_idx in range (len(thred)):
                tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                auroc_score += (-fpr+fprTemp)*tpr
                fprTemp = fpr
            auroc_score += fpr * tpr
            info['{}_auroc_{}'.format(item,score_type)] = auroc_score
            print('use time:{:.1f} second'.format(time.time()-t0))

    if args.save.find('ensemble')>0:
        torch.save(info,"{}/info_{}.pt".format(args.save,args.calibration))
    else:
        torch.save(info,"{}/info.pt".format(args.save))
    print(info)

def auprIn(args):
    score_type_list=['Msp','Msp_Tsc','Msp_MaxOdin']
    out_dataset_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN','Gaussian', 'Uniform']
    info=torch.load('{}/info.pt'.format(args.save))

    for score_type in score_type_list:
        in_score = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
        for item in out_dataset_list:
            out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item))
            thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
            precisionVec = []
            recallVec = []
            auprIn = 0.0
            recallTemp = 1.0
            for thred_idx in range (len(thred)):
                tp = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                fp = torch.sum(torch.sum(out_score >= thred[thred_idx])) / float(len(out_score))
                if tp + fp == 0: continue
                precision = tp / (tp + fp)
                recall = tp
                precisionVec.append(precision)
                recallVec.append(recall)
                auprIn += (recallTemp-recall)*precision
                recallTemp = recall
            auprIn += recall * precision
            info['{}_auprIn_{}'.format(item,score_type)] = auprIn
    torch.save(info,"{}/info.pt".format(args.save))
    print(info)

def auprOut(args):
    score_type_list=['Msp','Msp_Tsc','Msp_MaxOdin']
    out_dataset_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN','Gaussian', 'Uniform']
    info=torch.load('{}/info.pt'.format(args.save))

    for score_type in score_type_list:
        in_score = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
        for item in out_dataset_list:
            out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
            thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
            auprOut = 0.0
            recallTemp = 1.0
            precision=1.0
            for thred_idx in range (len(thred)):
                fp = torch.sum(torch.sum(in_score < thred[len(thred)-thred_idx-1])) / float(len(in_score))
                tp = torch.sum(torch.sum(out_score < thred[len(thred)-thred_idx-1])) / float(len(out_score))
                if tp + fp == 0: break
                precision = tp / (tp + fp)
                recall = tp
                auprOut += (recallTemp-recall)*precision
                recallTemp = recall
            auprOut += recallTemp * precision
            info['{}_auprOut_{}'.format(item,score_type)] = auprOut
    torch.save(info,"{}/info.pt".format(args.save))
    print(info)
    
def detection(args):
    score_type_list=['Msp','Msp_Tsc','Msp_MaxOdin']
    out_dataset_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN','Gaussian', 'Uniform']
    info=torch.load('{}/info.pt'.format(args.save))

    for score_type in score_type_list:
        in_score = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
        for item in out_dataset_list:
            out_score = torch.load('{}/score/{}_{}.pt'.format(args.save,item,score_type))
            thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
            detec = 1.0
            for thred_idx in range (len(thred)):
                tpr = torch.sum(torch.sum(in_score < thred[thred_idx])) / float(len(in_score))
                error2 = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                detec = np.minimum(detec, (tpr+error2)/2.0)
            info['{}_detec_{}'.format(item,score_type)] = detec
    torch.save(info,"{}/info.pt".format(args.save))
    print(info)
   
