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

def cal_metric(args,metric_type_list,out_datasets):
    if 'tnr95' in metric_type_list:
        tnr95(args,out_datasets)
    if 'auroc' in metric_type_list:
        auroc(args,out_datasets)
    # if 'auprIn' in metric_type_list:
    #     auprIn(args)
    # if 'auprOut' in metric_type_list:
    #     auprOut(args)
    # if 'detec' in metric_type_list:
    #     detection(args)
    return

def tnr95(args,out_datasets):
    if 'eval' in args.inferance:
        if args.save.find('ensemble')>0:
            info=torch.load('{}/eval_info_{}.pt'.format(args.save,args.calibration))
        else:
            info=torch.load('{}/eval_info.pt'.format(args.save))

        if args.save.find('ensemble')>0:
            score_type_list=['cali_Pred','cali_Pred_Tsc']
            for score_type in score_type_list:
                if(len(out_datasets)==1 and out_datasets[0]=='val'):
                    in_score = torch.sort(torch.load('{}/eval_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
                else:
                    in_score = torch.sort(torch.load('{}/eval_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
                for item in out_datasets:
                    out_score=torch.load('{}/eval_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                    info['{}_{}_tnr95_{}'.format(args.in_dataset,item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
                    
        score_type_list=['Pred','Pred_Tsc']
        for score_type in score_type_list:
            if(len(out_datasets)==1 and out_datasets[0]=='val'):
                soft_max = torch.load('{}/eval_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type))
            else:
                soft_max = torch.load('{}/eval_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
            in_score = torch.sort(torch.sum(soft_max*torch.log(soft_max),dim=1))[0]
            for item in out_datasets:
                soft_max = torch.load('{}/eval_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
                info['{}_{}_tnr95_{}'.format(args.in_dataset,item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
            
        if args.save.find('ensemble')>0:
            torch.save(info,"{}/eval_info_{}.pt".format(args.save,args.calibration))
        else:
            torch.save(info,"{}/eval_info.pt".format(args.save))
        print(info)

        
    if 'ood' in args.inferance:
        if args.save.find('ensemble')>0:
            info=torch.load('{}/ood_info_{}.pt'.format(args.save,args.calibration))
        else:
            info=torch.load('{}/ood_info.pt'.format(args.save))

        if args.save.find('ensemble')>0:
            score_type_list=['cali_Pred','cali_Pred_Tsc']
            for score_type in score_type_list:
                if(len(out_datasets)==1 and out_datasets[0]=='val'):
                    in_score = torch.sort(torch.load('{}/ood_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
                else:
                    in_score = torch.sort(torch.load('{}/ood_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type)))[0]
                for item in out_datasets:
                    out_score=torch.load('{}/ood_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                    info['{}_{}_tnr95_{}'.format(args.in_dataset,item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
                    
        score_type_list=['Pred','Pred_Tsc']
        for score_type in score_type_list:
            if(len(out_datasets)==1 and out_datasets[0]=='val'):
                soft_max = torch.load('{}/ood_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type))
            else:
                soft_max = torch.load('{}/ood_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
            in_score = torch.sort(torch.sum(soft_max*torch.log(soft_max),dim=1))[0]
            for item in out_datasets:
                soft_max = torch.load('{}/ood_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                out_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
                info['{}_{}_tnr95_{}'.format(args.in_dataset,item,score_type)] = 1-torch.sum(torch.sum(out_score>in_score[int(round(len(in_score)*0.05))]))/float(len(out_score))
            
        if args.save.find('ensemble')>0:
            torch.save(info,"{}/ood_info_{}.pt".format(args.save,args.calibration))
        else:
            torch.save(info,"{}/ood_info.pt".format(args.save))
        print(info)

def auroc(args,out_datasets):
    t0 = time.time()
    if 'eval' in args.inferance:
        if args.save.find('ensemble')>0:
            info=torch.load('{}/eval_info_{}.pt'.format(args.save,args.calibration))
        else:
            info=torch.load('{}/eval_info.pt'.format(args.save))

        if args.save.find('ensemble')>0:
            score_type_list=['cali_Pred','cali_Pred_Tsc']
            for score_type in score_type_list:
                for item in out_datasets:
                    if(len(out_datasets)==1 and out_datasets[0]=='val'):
                        in_score = torch.load('{}/eval_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type))
                    else:
                        in_score = torch.load('{}/eval_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
                    print('\ncal auroc:{}/eval_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                    out_score = torch.load('{}/eval_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                    thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                    auroc_score = 0.0
                    fprTemp = 1.0
                    for thred_idx in range (len(thred)):
                        tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                        fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                        auroc_score += (-fpr+fprTemp)*tpr
                        fprTemp = fpr
                    auroc_score += fpr * tpr
                    info['{}_{}_auroc_{}'.format(args.in_dataset,item,score_type)] = auroc_score
                    print('use time:{:.1f} second'.format(time.time()-t0))

        score_type_list=['Pred','Pred_Tsc']
        for score_type in score_type_list:
            if(len(out_datasets)==1 and out_datasets[0]=='val'):
                soft_max = torch.load('{}/eval_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type))
            else:
                soft_max = torch.load('{}/eval_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
            in_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            for item in out_datasets:
                print('\ncal auroc:{}/eval_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                soft_max = torch.load('{}/eval_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
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
                info['{}_{}_auroc_{}'.format(args.in_dataset,item,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))
        if args.save.find('ensemble')>0:
            torch.save(info,"{}/eval_info_{}.pt".format(args.save,args.calibration))
        else:
            torch.save(info,"{}/eval_info.pt".format(args.save))
        print(info)

    if 'ood' in args.inferance:
        if args.save.find('ensemble')>0:
                info=torch.load('{}/ood_info_{}.pt'.format(args.save,args.calibration))
        else:
            info=torch.load('{}/ood_info.pt'.format(args.save))

        if args.save.find('ensemble')>0:
            score_type_list=['cali_Pred','cali_Pred_Tsc']
            for score_type in score_type_list:
                for item in out_datasets:
                    if(len(out_datasets)==1 and out_datasets[0]=='val'):
                        in_score = torch.load('{}/ood_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type))
                    else:
                        in_score = torch.load('{}/ood_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
                    print('\ncal auroc:{}/ood_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                    out_score = torch.load('{}/ood_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                    thred = torch.sort(torch.cat((in_score,out_score),dim=0))[0]
                    auroc_score = 0.0
                    fprTemp = 1.0
                    for thred_idx in range (len(thred)):
                        tpr = torch.sum(torch.sum(in_score >= thred[thred_idx])) / float(len(in_score))
                        fpr = torch.sum(torch.sum(out_score > thred[thred_idx])) / float(len(out_score))
                        auroc_score += (-fpr+fprTemp)*tpr
                        fprTemp = fpr
                    auroc_score += fpr * tpr
                    info['{}_{}_auroc_{}'.format(args.in_dataset,item,score_type)] = auroc_score
                    print('use time:{:.1f} second'.format(time.time()-t0))

        score_type_list=['Pred','Pred_Tsc']
        for score_type in score_type_list:
            if(len(out_datasets)==1 and out_datasets[0]=='val'):
                soft_max = torch.load('{}/ood_score/{}_val_in_{}.pt'.format(args.save,args.in_dataset,score_type))
            else:
                soft_max = torch.load('{}/ood_score/{}_{}.pt'.format(args.save,args.in_dataset,score_type))
            in_score = torch.sum(soft_max*torch.log(soft_max),dim=1)
            for item in out_datasets:
                print('\ncal auroc:{}/ood_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
                soft_max = torch.load('{}/ood_score/{}_{}_{}.pt'.format(args.save,args.in_dataset,item,score_type))
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
                info['{}_{}_auroc_{}'.format(args.in_dataset,item,score_type)] = auroc_score
                print('use time:{:.1f} second'.format(time.time()-t0))
        if args.save.find('ensemble')>0:
            torch.save(info,"{}/ood_info_{}.pt".format(args.save,args.calibration))
        else:
            torch.save(info,"{}/ood_info.pt".format(args.save))
        print(info)
