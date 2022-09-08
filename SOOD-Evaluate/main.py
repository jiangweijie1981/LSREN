from numpy.lib import utils
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
from evaluate import *


if __name__=='__main__':
    if not os.path.exists('cifar10_id_lists.pt') or not os.path.exists('cifar100_id_lists.pt'):
        generate_id_list()
    args=Config().data
    out_datasets=['out','val','Imagenet','Imagenet_resize','LSUN','LSUN_resize','iSUN']
    for s in range (args.seed_end-args.seed_begin+1):
        args.seed=args.seed_begin+s
        set_seed(args.seed)
        smooth_list=get_smooth_list(args)
        print(smooth_list)
        if args.train_type=='LSR':
            for i in range(len(smooth_list)):
                args.smooth=smooth_list[i]
                args.save = '../models/{}_seed{}_{}_{}-{}/{}_esbl{}_range{}/epochs{}_s{:.6f}'.format(
                    args.net,args.seed,args.in_dataset,args.dataset_id,args.samples_pre_class_num,
                    args.train_type,args.ensemble_num,args.smooth_range,args.epochs,args.smooth)
                if 'train' in args.excute_list:
                    train(args)
                if 'evaluate' in args.excute_list:
                    evaluate(args)
                if 'cal_magnitude' in args.excute_list:
                    cal_magnitude(args)
                if 'cal_score' in args.excute_list:
                    cal_score(args)
                if 'cal_metric' in args.excute_list:
                    cal_metric(args,['tnr95','auroc'],out_datasets)
                if 'print_info' in args.excute_list:
                    print_info(args)
