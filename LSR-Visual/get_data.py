from random import sample
from numpy.lib import utils
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import time
import torch.utils.data

from torchvision.transforms.transforms import Grayscale, Resize

def get_dataloader(args,dataset_type):
    args.train_batch_size = args.train_batch_size*torch.cuda.device_count()
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}

    if args.dataset_name=='mnist':
        id_list=args.train_class_list
        mean,std=cal_mean_std(args,id_list)
        tr_transform=transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        te_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        if dataset_type=='train':
            fullset=torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True,transform=tr_transform)
            dataset=get_split_set(fullset,id_list,args.proportion,args)
            print('{},load train dataset:mnist,class_id_list:{}, length:{}'.format(time.asctime(time.localtime(time.time())),id_list,len(dataset)))
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.MNIST(root=args.dataset_path, train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,id_list,1,args)
            print('{},load test dataset:mnist,class_id_list:{}, length:{}'.format(time.asctime(time.localtime(time.time())),id_list,len(dataset)))
            return torch.utils.data.DataLoader(dataset,batch_size=args.infer_batch_size, shuffle=False, **kwargs)
        if dataset_type=='infer':
            fullset=torchvision.datasets.MNIST(root=args.dataset_path, train=False, download=True,transform=te_transform)
            print('{},load infer dataset:mnist, length:{}'.format(time.asctime(time.localtime(time.time())),len(fullset)))
            return torch.utils.data.DataLoader(fullset,batch_size=args.infer_batch_size, shuffle=False, **kwargs)
        elif dataset_type=='noise':
            images=torch.load('{}/{}/images_tensor.pt'.format(args.dataset_root,'Gaussian28'))[:1000]
            images[:,0]=(images[:,0]-mean[0])/std[0]
            out_dataloader=[]
            for i in range(int(len(images)/args.infer_batch_size)):
                batch_data=images[i*args.infer_batch_size:(i+1)*args.infer_batch_size]
                out_dataloader.append((batch_data,i))
            # images=torch.load('{}/{}/images_tensor.pt'.format(args.dataset_root,'Uniform28'))[:500]
            # images[:,0]=(images[:,0]-mean[0])/std[0]
            # # out_dataloader=[]
            # for i in range(int(len(images)/args.infer_batch_size)):
            #     batch_data=images[i*args.infer_batch_size:(i+1)*args.infer_batch_size]
            #     out_dataloader.append((batch_data,i))
            return out_dataloader  
def get_split_set(src,class_id_list,proportion,args):
    # print('test:')
    # print(id_list)
    image_path = []
    image_label = []
    if proportion<1:
        full_sample_id_list=torch.load('{}/full_sample_id_shuffle_list.pt'.format(args.dataset_config_path))
        sample_length=int(len(src.data)*proportion)
        sample_id_list=full_sample_id_list[:sample_length]
        for i in range(len(src.data)):
            if (int(src.targets[i]) in class_id_list) and (i in sample_id_list):
                image_path.append(src.data[i])
                image_label.append(class_id_list.index(src.targets[i]))
    else:
        for i in range(len(src.data)):
            if (int(src.targets[i]) in class_id_list):
                image_path.append(src.data[i])
                image_label.append(class_id_list.index(src.targets[i]))
    src.data = image_path
    src.targets = image_label
    return src

def cal_mean_std(args,id_list):
    if os.path.exists('{}/mean_std.pt'.format(args.dataset_config_path)):
        mean_std=torch.load('{}/mean_std.pt'.format(args.dataset_config_path))
        return mean_std['mean'],mean_std['std']
    print('{},cal mean&std on {}, class id_list:{}'.format(time.asctime(time.localtime(time.time())),args.dataset_name,id_list))
    if args.dataset_name=='cifar6' or args.dataset_name=='cifar10':
        fullset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    if args.dataset_name=='cifar80'  or args.dataset_name=='cifar100':
        fullset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    if args.dataset_name=='mnist':
        fullset = torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    dataset=get_split_set(fullset,id_list,args.proportion,args)
    dataLoader=torch.utils.data.DataLoader(dataset,batch_size=100, shuffle=False)
    for idx, (data, __) in enumerate(dataLoader):
        if idx==0:
            total=data.sum(dim=(0,2,3))
        else:
            total+=data.sum(dim=(0,2,3))
    mean=total/(len(dataset)*32*32)
    print('mean:{}'.format(mean))
    
    for idx, (data, __) in enumerate(dataLoader):
        if idx==0:
            temp=(torch.unsqueeze(mean,dim=0)).expand(data.size(0),data.size(1))
            temp=(torch.unsqueeze(temp,dim=2)).expand(data.size(0),data.size(1),data.size(2))
            temp=(torch.unsqueeze(temp,dim=3)).expand(data.size(0),data.size(1),data.size(2),data.size(3))
            total=(data-temp).pow(2).sum(dim=(0,2,3))
        else:
            temp=(torch.unsqueeze(mean,dim=0)).expand(data.size(0),data.size(1))
            temp=(torch.unsqueeze(temp,dim=2)).expand(data.size(0),data.size(1),data.size(2))
            temp=(torch.unsqueeze(temp,dim=3)).expand(data.size(0),data.size(1),data.size(2),data.size(3))
            total+=(data-temp).pow(2).sum(dim=(0,2,3))
    std=torch.sqrt(total/(len(dataset)*32*32-1))
    print('std:{}'.format(std))
    mean_std=dict()
    mean_std['mean']=mean
    mean_std['std']=std
    torch.save(mean_std,'{}/mean_std.pt'.format(args.dataset_config_path))
    return mean,std
