import argparse
import torch
import ast

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Pytorch OOD Detection')

        self.parser.add_argument('--dataset_root', type=str, default='../../Datasets/') # Default
        self.parser.add_argument('--dataset_path', type=str, default='no_set') # Runtime
        self.parser.add_argument('--dataset_config_path', type=str, default='no_set') # Runtime
        self.parser.add_argument('--output_path', type=str, default='no_set') # Runtime

        self.parser.add_argument('--dataset_name', type=str, default='mnist',\
            choices=('cifar80', 'cifar6','cifar10','cifar100','mnist'))# Parameters
        self.parser.add_argument('--dataset_id', type=int, default=0)# Parameters
        self.parser.add_argument('--class_num', type=int, default=100)# Runtime
        self.parser.add_argument('--proportion', type=float, default=1.0)
        self.parser.add_argument('--dropout', type=float, default=0.5)

        self.parser.add_argument('--seed', type=int, default=0) # Runtime
        self.parser.add_argument('--seed_begin', type=int, default=0) # Parameters
        self.parser.add_argument('--seed_end', type=int, default=0) # Parameters

        self.parser.add_argument('--opt', type=str, default='sgd',choices=('sgd', 'adam', 'rmsprop')) # Default
        self.parser.add_argument('--net', type=str, default='lenet',choices=('resnet','resnet_cos','dense_cos',\
            'dense','resnet_dropout34','resnet_dropout50','lenet','lenet_BN','lenet_Dropout','resnet_nor')) # Parameters

        self.parser.add_argument('--num_workers', type=int, default=0) # Parameters
        self.parser.add_argument('--train_batch_size', type=int, default=64) # Runtime
        self.parser.add_argument('--decay', type=float, default=0.0001) # Runtime
        self.parser.add_argument('--epochs', type=int, default=20) # Runtime
        self.parser.add_argument('--infer_batch_size', type=int, default=100) # Runtime
        self.parser.add_argument('--multi_infer_num', type=int, default=1) # Runtime

        self.parser.add_argument('--esbl_num', type=int, default=1) # Parameters
        self.parser.add_argument('--esbl_epochs', type=int, default=100) # Parameters Multi mode
        self.parser.add_argument('--smoothness', type=float, default=0.0) # Parameters Single mode
        self.parser.add_argument('--smoothness_range', type=float, default=0.5)# Default
        self.parser.add_argument('--temperature', default = 1000, type = int) # Parameters Multi mode
        self.parser.add_argument('--fea_dim', default = 512, type = int) # Runtime
        self.parser.add_argument('--esbl_fea_dim', default = 1000, type = int) # Default
        self.parser.add_argument('--esbl_lr', type=float, default=0.1) # Default
        self.parser.add_argument('--esbl_rec_coef', type=float, default=0.01) # Parameters Multi mode
        self.parser.add_argument('--esbl_name', type=str, default='') # Runtime

        self.parser.add_argument('--esbl_types_list', type=str, nargs='+', default='nor mlp cos')
        self.parser.add_argument('--eval_types_list', type=str, nargs='+', default='pred scale')
        self.parser.add_argument('--train_class_list', type=int, nargs='+', default=[0,1])
        self.parser.add_argument('--visual_class_list', type=int, nargs='+', default=[0,1])

        # self.parser.register('type','bool',self.str2boolstr2bool)
        self.parser.add_argument('--mixup',  default='False',type=self.str2bool)
        self.parser.add_argument('--bn',  default='False',type=self.str2bool)
        self.parser.add_argument('--osbn',  default='False',type=self.str2bool)


        self.parser.set_defaults(argument=True)
        self.data= self.parser.parse_args()

    def str2bool(self,v):
        if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'False','false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
