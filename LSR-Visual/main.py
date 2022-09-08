from datetime import time
import os
from visual_fea import visual
from utils import dataset_config,set_seed
# from configs import Config
from get_data import *
from train import *
import csv
import torch


if __name__=='__main__':
    args=Config().data
    args.train_class_list=[0,1,2]
    # args.train_class_list=list(range(10))
    gt=[torch.tensor([0.6,0.4,0]),torch.tensor([0.4,0.6,0]),torch.tensor([0,0,1])]
    args.visual_class_list=[5,6,7,1,10]
    # args.visual_class_list=list(range(11))
    set_dataset_path(args)
    dataset_config(args)
    args.output_path = '{}/{}'.format(args.dataset_config_path,args.net)
    train(args,gt)
    # visual(args)
