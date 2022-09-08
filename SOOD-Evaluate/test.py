import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from configs import *
from get_data import *
from utils import *
import torch.nn.functional as F



args=Config().data
args.class_num=6
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

            model_path="{}/latest-{}.pth".format(args.save,args.epochs)
            if args.net=='resnet':
                import net.resnet as resnet
                net = resnet.ResNet34(num_c=args.class_num)
            elif args.net=='resnet_cos':
                import net.resnet_cos as resnet
                net = resnet.ResNet34(num_c=args.class_num)
            elif args.net=='resnet_inn':
                import net.resnet_inn as resnet
                net = resnet.ResNet34(num_c=args.class_num)
            net.load_state_dict(torch.load(model_path,map_location='cpu')['state_dict'])


            torch.save({'state_dict': net.state_dict()}, '{}/latest-{}.pth'.format(args.save,100),_use_new_zipfile_serialization=False)

            print('check {}'.format(args.smooth))