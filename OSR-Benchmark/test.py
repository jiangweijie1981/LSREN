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
import csv



if __name__=='__main__':
    args=Config().data
    set_seed(args.seed)
    if args.in_dataset=='tiny_imagenet':
        args.class_num=20
    elif args.in_dataset=='cifar4':
        args.class_num=4
    else:
        args.class_num=6
    model_list=get_ensemble_list(args)
    print(model_list)

    save_path = '../models/{}_seed{}_{}/{}_esbl{}_range{}/ensemble_epochs{}'.format(
        args.net,args.seed,args.in_dataset,
        args.train_type,args.ensemble_num,args.smooth_range,args.epochs)


    with open('{}/train_acc_seed{}.txt'.format(save_path,args.seed),'wt') as f:
        for item in model_list:
            csv_path='{}/train.csv'.format(item)
            with open(csv_path,'r') as f1:
                lines=csv.reader(f1)
                for i,line in enumerate(lines):
                    if i==7899:
                        print('{} '.format(line[3]),file=f)

    
