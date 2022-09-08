import argparse
import torch
import ast

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Pytorch Ensemble Detection')
        self.parser.add_argument('--in_dataset', type=str, default='cifar80',choices=('cifar6','cifar60','cifar80'))
        self.parser.add_argument('--train_type', type=str, default='LSR',choices=('LSR', 'ILSR'))
        self.parser.add_argument('--opt', type=str, default='sgd',choices=('sgd', 'adam', 'rmsprop'))
        self.parser.add_argument('--net', type=str, default='resnet',choices=('resnet','resnet_cos','resnet_inn','resnet_euc','resnet_dropout34','resnet_dropout50'))
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--samples_pre_class_num', type=int, default=5000)
        self.parser.add_argument('--dataset_id', type=int, default=0)

        self.parser.add_argument('--seed', type=int, default=0)
        self.parser.add_argument('--seed_begin', type=int, default=0)
        self.parser.add_argument('--seed_end', type=int, default=0)    
        self.parser.add_argument('--ensemble_num', type=int, default=1)
        self.parser.add_argument('--save')
        self.parser.add_argument('--smooth', type=float, default=0.0)
        self.parser.add_argument('--smooth_range', type=float, default=0.5)
        self.parser.add_argument('--calibration', type=float, default=1.0)
        self.parser.add_argument('--calibration_interval', type=float, default=0.05)

        self.parser.add_argument('--train_batch_size', type=int, default=64)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--decay', type=float, default=0.0001)
        self.parser.add_argument('--class_num', type=int, default=100)

        self.parser.add_argument('--magnitude', default=0.0, type=float,help='perturbation magnitude')
        self.parser.add_argument('--temperature', default=1000, type=float,help='temperature scaling')
        self.parser.add_argument('--ood_batch_size', default = 100, type = int)
        self.parser.add_argument('--ood_begin_index', type=int,default=1000)
        self.parser.add_argument('--inferance', type=str, nargs='+', default='ood')

        self.parser.add_argument('--excute_list', type=str,nargs='+',\
            default='train cal_magnitude cal_score cal_metric print_info cal_acc wr')

        self.parser.set_defaults(argument=True)
        self.data= self.parser.parse_args()

    def str2bool(self,v):
        if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'False','false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
