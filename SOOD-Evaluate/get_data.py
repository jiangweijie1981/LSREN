from numpy.lib import utils
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import os

def get_dataloader(args,dataset_type):
    args.train_batch_size = args.train_batch_size*torch.cuda.device_count()
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}

    if args.in_dataset=='cifar6':
        args.class_num=6
        id_lists=torch.load('cifar10_id_lists.pt')
        id_list=id_lists[args.dataset_id]
        mean,std=cal_mean_std(args,id_list[:6])
        tr_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        te_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        if dataset_type=='train':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=tr_transform)
            dataset=get_split_set_train(fullset,args,id_list[:6])
            print(id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:6])
            print(id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)
        if dataset_type=='in':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:6])
            print(id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[6:])
            print(id_list[6:])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val_in':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=te_transform)
            dataset=get_split_set_train(fullset,args,id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            return torch.utils.data.DataLoader(fullset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
    if args.in_dataset=='cifar60':
        args.class_num=60
        id_lists=torch.load('cifar100_id_lists.pt')
        id_list=id_lists[args.dataset_id]
        mean,std=cal_mean_std(args,id_list[:60])
        tr_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        te_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        if dataset_type=='train':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=tr_transform)
            dataset=get_split_set_train(fullset,args,id_list[:60])
            print(id_list[:60])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:60])
            print(id_list[:60])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)
        if dataset_type=='in':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:60])
            print(id_list[:60])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[60:])
            print(id_list[60:])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            return torch.utils.data.DataLoader(fullset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
    if args.in_dataset=='cifar80':
        args.class_num=80
        id_lists=torch.load('cifar100_id_lists.pt')
        id_list=id_lists[args.dataset_id]
        mean,std=cal_mean_std(args,id_list[:80])
        tr_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        te_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        if dataset_type=='train':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=tr_transform)
            dataset=get_split_set_train(fullset,args,id_list[:80])
            print(id_list[:80])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:80])
            print(id_list[:80])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)
        if dataset_type=='in':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:80])
            print(id_list[:80])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[80:])
            print(id_list[80:])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val_in':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=te_transform)
            dataset=get_split_set_train(fullset,args,id_list[:80])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            return torch.utils.data.DataLoader(fullset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)



def get_split_set_train(src,args,id_list):
    print(id_list)
    samples_sum=dict()
    for i in id_list:
        samples_sum[i]=0
    samples_max=args.samples_pre_class_num

    image_path = []
    image_label = []
    for i in range(len(src.data)):
        if (int(src.targets[i]) in id_list) and (samples_sum[src.targets[i]]<samples_max):
            image_path.append(src.data[i])
            image_label.append(id_list.index(src.targets[i]))
            samples_sum[src.targets[i]]+=1
    src.data = image_path
    src.targets = image_label
    return src

def get_split_set(src,args,id_list):
    print(id_list)

    image_path = []
    image_label = []
    for i in range(len(src.data)):
        if int(src.targets[i]) in id_list:
            image_path.append(src.data[i])
            image_label.append(id_list.index(src.targets[i]))
    src.data = image_path
    src.targets = image_label

    return src

def cal_mean_std(args,id_list):
    if os.path.exists('{}/mean_std.pt'.format(args.save)):
        mean_std=torch.load('{}/mean_std.pt'.format(args.save))
        return mean_std['mean'],mean_std['std']
    if args.in_dataset=='cifar6':
        fullset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    if args.in_dataset=='cifar60':
        fullset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    if args.in_dataset=='cifar80':
        fullset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    dataset=get_split_set(fullset,args,id_list)
    dataLoader=torch.utils.data.DataLoader(dataset,batch_size=100, shuffle=False)
    for idx, (data, __) in enumerate(dataLoader):
        if idx==0:
            total=data.sum(dim=(0,2,3))
        else:
            total+=data.sum(dim=(0,2,3))
    mean=total/(len(dataset)*32*32)
    print(mean)
    
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
    print(std)
    mean_std=dict()
    mean_std['mean']=mean
    mean_std['std']=std
    torch.save(mean_std,'{}/mean_std.pt'.format(args.save))
    return mean,std

def generate_id_list():
    id_lists=[]
    for i in range(10):
        np.random.seed(i)
        id_list=list(range(10))
        np.random.shuffle(id_list)
        id_lists.append(id_list)
    torch.save(id_lists,'cifar10_id_lists.pt')
    id_lists=[]
    for i in range(10):
        np.random.seed(i)
        id_list=list(range(100))
        np.random.shuffle(id_list)
        id_lists.append(id_list)
    torch.save(id_lists,'cifar100_id_lists.pt')