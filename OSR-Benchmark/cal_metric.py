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
    return

def tnr95(args):
    if args.save.find('ensemble')>0:
        info=torch.load('{}/info_{}.pt'.format(args.save,args.calibration))
    else:
        info=torch.load('{}/info.pt'.format(args.save))

    if args.save.find('ensemble')>0:
        score_type_list=['cali_Pred','cali_Pred_Tsc']
        for score_type in score_type_list:
            in_score = torch.sort(torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
            if args.in_dataset.find('4')>0:
                out_score=torch.load('{}/score/{}_out10_{}.pt'.format(args.save,args.in_dataset,score_type))
                info['{}_out10_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
                out_score=torch.load('{}/score/{}_out50_{}.pt'.format(args.save,args.in_dataset,score_type))
                info['{}_out50_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
            else:
                out_score=torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
                info['{}_out_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
            out_score=torch.load('{}/score/{}_val_{}.pt'.format(args.save,args.in_dataset,score_type))
            info['{}_val_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
    score_type_list=['Pred','Pred_Tsc']
    for score_type in score_type_list:
        soft_max = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
        in_score = torch.sort(torch.sum(soft_max*torch.log(soft_max),dim=1))[0]
        if args.in_dataset.find('4')>0:
            soft_max = torch.load('{}/score/{}_out10_{}.pt'.format(args.save,args.in_dataset,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            info['{}_out10_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
            soft_max = torch.load('{}/score/{}_out50_{}.pt'.format(args.save,args.in_dataset,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            info['{}_out50_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
        else:
            soft_max = torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
            out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            info['{}_out_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
        soft_max = torch.load('{}/score/{}_val_{}.pt'.format(args.save,args.in_dataset,score_type))
        out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
        info['{}_val_tnr95_{}'.format(args.in_dataset,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
        
    if args.save.find('ensemble')>0:
        torch.save(info,"{}/info_{}.pt".format(args.save,args.calibration))
    else:
        torch.save(info,"{}/info.pt".format(args.save))
    print(info)

def auroc(args):
    t0 = time.time()

    if args.save.find('ensemble')>0:
        info=torch.load('{}/info_{}.pt'.format(args.save,args.calibration))
    else:
        info=torch.load('{}/info.pt'.format(args.save))
    
    if args.save.find('ensemble')>0:
        score_type_list=['cali_Pred','cali_Pred_Tsc']
        for score_type in score_type_list:
            if args.in_dataset.find('4')>0:
                in_score = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
                print('\ncal auroc:{}/score/{}_out10_{}.pt'.format(args.save,args.in_dataset,score_type))
                out_score = torch.load('{}/score/{}_out10_{}.pt'.format(args.save,args.in_dataset,score_type))
                thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                auroc_score = 0.0
                fprTemp = 1.0
                for thred_idx in range (len(thred)):
                    tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                    fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                    auroc_score += (-fpr+fprTemp)*tpr
                    fprTemp = fpr
                auroc_score += fpr * tpr
                info['{}_out10_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))

                print('\ncal auroc:{}/score/{}_out50_{}.pt'.format(args.save,args.in_dataset,score_type))
                out_score = torch.load('{}/score/{}_out50_{}.pt'.format(args.save,args.in_dataset,score_type))
                thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                auroc_score = 0.0
                fprTemp = 1.0
                for thred_idx in range (len(thred)):
                    tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                    fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                    auroc_score += (-fpr+fprTemp)*tpr
                    fprTemp = fpr
                auroc_score += fpr * tpr
                info['{}_out50_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))
            else:
                in_score = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
                print('\ncal auroc:{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
                out_score = torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
                thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                auroc_score = 0.0
                fprTemp = 1.0
                for thred_idx in range (len(thred)):
                    tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                    fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                    auroc_score += (-fpr+fprTemp)*tpr
                    fprTemp = fpr
                auroc_score += fpr * tpr
                info['{}_out_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))
            in_score = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
            print('\ncal auroc:{}/score/{}_val_{}.pt'.format(args.save,args.in_dataset,score_type))
            out_score = torch.load('{}/score/{}_val_{}.pt'.format(args.save,args.in_dataset,score_type))
            thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
            auroc_score = 0.0
            fprTemp = 1.0
            for thred_idx in range (len(thred)):
                tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                auroc_score += (-fpr+fprTemp)*tpr
                fprTemp = fpr
            auroc_score += fpr * tpr
            info['{}_val_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
            print('use time:{:.1f} second'.format(time.time()-t0))
    score_type_list=['Pred','Pred_Tsc']
    for score_type in score_type_list:
        soft_max = torch.load('{}/score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
        in_score = torch.sum(soft_max*torch.log(soft_max),dim=1)

        if args.in_dataset.find('4')>0:
            print('\ncal auroc:{}/score/{}_out10_{}.pt'.format(args.save,args.in_dataset,score_type))
            soft_max = torch.load('{}/score/{}_out10_{}.pt'.format(args.save,args.in_dataset,score_type))
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
            info['{}_out10_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
            print('use time:{:.1f} second'.format(time.time()-t0))

            print('\ncal auroc:{}/score/{}_out50_{}.pt'.format(args.save,args.in_dataset,score_type))
            soft_max = torch.load('{}/score/{}_out50_{}.pt'.format(args.save,args.in_dataset,score_type))
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
            info['{}_out50_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
            print('use time:{:.1f} second'.format(time.time()-t0))
        else:
            print('\ncal auroc:{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
            soft_max = torch.load('{}/score/{}_out_{}.pt'.format(args.save,args.in_dataset,score_type))
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
            info['{}_out_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
            print('use time:{:.1f} second'.format(time.time()-t0))
        print('\ncal auroc:{}/score/{}_val_{}.pt'.format(args.save,args.in_dataset,score_type))
        soft_max = torch.load('{}/score/{}_val_{}.pt'.format(args.save,args.in_dataset,score_type))
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
        info['{}_val_auroc_{}'.format(args.in_dataset,score_type)] = auroc_score
        print('use time:{:.1f} second'.format(time.time()-t0))
    if args.save.find('ensemble')>0:
        torch.save(info,"{}/info_{}.pt".format(args.save,args.calibration))
    else:
        torch.save(info,"{}/info.pt".format(args.save))
    print(info)
