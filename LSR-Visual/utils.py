from math import inf
import numpy as np
import random
# import openpyxl
import torch
import os
import torch.utils.data.dataset
from configs import *
from get_data import *
from torch.autograd import Variable


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True

def set_dataset_path(args):
    if args.dataset_name=='cifar10':
        args.dataset_path='{}/cifar10'.format(args.dataset_root)
        args.class_num=10
    if args.dataset_name=='cifar6':
        args.dataset_path='{}/cifar10'.format(args.dataset_root)
        args.class_num=6
    if args.dataset_name=='cifar100':
        args.dataset_path='{}/cifar100'.format(args.dataset_root)
        args.class_num=100
    if args.dataset_name=='cifar80':
        args.dataset_path='{}/cifar100'.format(args.dataset_root)
        args.class_num=80
    if args.dataset_name=='mnist':
        args.dataset_path='{}/mnist'.format(args.dataset_root)
        args.class_num=len(args.train_class_list)
    if not os.path.exists('../outputs'):
        os.mkdir('../outputs')
    args.dataset_config_path='../outputs/{}-{}'.format(args.dataset_name,args.dataset_id)

def dataset_config(args):
    if not os.path.exists(args.dataset_config_path):
        os.mkdir(args.dataset_config_path)

    if not os.path.exists('{}/full_sample_id_shuffle_list.pt'.format(args.dataset_config_path)):
        fullset=torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True)
        np.random.seed(args.dataset_id)
        full_id_list=list(range(len(fullset)))
        np.random.shuffle(full_id_list)
        torch.save(full_id_list,'{}/full_sample_id_shuffle_list.pt'.format(args.dataset_config_path))
    else:
        full_id_list=torch.load('{}/full_sample_id_shuffle_list.pt'.format(args.dataset_config_path))



def load_net(args):
    if args.net=='lenet':
        import lenet as lenet
        net = lenet.LeNet(class_num=args.class_num)
    elif args.net=='lenet_BN':
        import lenet as lenet
        net = lenet.LeNet_BN()
    elif args.net=='lenet_Dropout':
        import lenet as lenet
        net = lenet.LeNet_Dropout(args.dropout)
    model_path='{}/models/seed{}_epochs{}_s{:.6f}_proportion{}.pth'.format(args.output_path,args.seed,args.epochs,args.smoothness,args.proportion)
    net.load_state_dict(torch.load(model_path)['state_dict'])
    if torch.cuda.device_count()>1:
        net=torch.nn.DataParallel(net)
    return net.cuda()

def get_net(args):
    if args.net=='lenet':
        import lenet as lenet
        net = lenet.LeNet(class_num=args.class_num)
    elif args.net=='lenet_BN':
        import lenet as lenet
        net = lenet.LeNet_BN()
    elif args.net=='lenet_Dropout':
        import lenet as lenet
        net = lenet.LeNet_Dropout(args.dropout)
    if torch.cuda.device_count()>1:
        net=torch.nn.DataParallel(net)
    return net.cuda()

