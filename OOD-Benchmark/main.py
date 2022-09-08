import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
from configs import *
from get_data import *
from utils import *
from cal_magnitude import *
from cal_score import *
from cal_metric import *
from train import *


if __name__=='__main__':
    args=Config().data
    set_seed(args.seed)
    smooth_list=get_smooth_list(args)
    print(smooth_list)
    if args.train_type=='LSR':
        for i in range(len(smooth_list)):
            args.smooth=smooth_list[i]
            args.save = '../models/{}_seed{}_{}_{}{}/{}_esbl{}_range{}/epochs{}_s{:.6f}'.format(
                args.net,args.seed,args.in_dataset,args.noise_type,args.noise_ratio,
                args.train_type,args.ensemble_num,args.smooth_range,args.epochs,args.smooth)
            if 'train' in args.excute_list:
                train(args)
            if 'cal_magnitude' in args.excute_list:
                cal_magnitude(args)
            if 'cal_score' in args.excute_list:
                cal_score(args)
            if 'cal_metric' in args.excute_list:
                cal_metric(args,['tnr95','auroc'])
            if 'print_info' in args.excute_list:
                print_info(args)
            if 'top_acc' in args.excute_list:
                get_top_acc(args)

    if args.train_type=='ILSR':
        for i in range(len(args.penalty_list)):
            args.penalty=args.penalty_list[i]
            for j in range(len(smooth_list)):
                args.smooth=smooth_list[j]
                args.save = '../models/{}_seed{}_{}_{}{}/{}_esbl{}_range{}_penalty{}/epochs{}_s{:.6f}'.format(
                    args.net,args.seed,args.in_dataset,args.noise_type,args.noise_ratio,
                    args.train_type,args.ensemble_num,args.smooth_range,args.penalty,args.epochs,args.smooth)
                if 'train' in args.excute_list:
                    train(args)
                if 'cal_magnitude' in args.excute_list:
                    cal_magnitude(args)
                if 'cal_score' in args.excute_list:
                    cal_score(args)
                if 'cal_metric' in args.excute_list:
                    cal_metric(args,['tnr95','auroc'])
                if 'print_info' in args.excute_list:
                    print_info(args)
                if 'top_acc' in args.excute_list:
                    get_top_acc(args)
                if 'plt' in args.excute_list:
                    plt_fig(args)