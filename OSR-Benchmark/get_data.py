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
        # split_list=[[0,1,2,4,5,9],[0,3,5,7,8,9],[0,1,5,6,7,8],[3,4,5,7,8,9],[0,1,2,3,7,8]]
        id_list=[0,1,2,3,4,5,6,7,8,9]
        np.random.seed(args.seed)
        np.random.shuffle(id_list)
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
            dataset=get_split_set(fullset,args,id_list[:6])
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
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            # dataset=get_split_set(fullset,args,id_list[6:])
            # print(id_list[6:])
            return torch.utils.data.DataLoader(fullset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    if args.in_dataset=='svhn6':
        args.class_num=6
        id_list=[0,1,2,3,4,5,6,7,8,9]
        np.random.seed(args.seed)
        np.random.shuffle(id_list)
        mean,std=cal_mean_std(args,id_list[:6])
        tr_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        te_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        if dataset_type=='train':
            fullset=torchvision.datasets.SVHN(root='../data',  download=True,split='train', transform=tr_transform)
            dataset=get_split_set(fullset,args,id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.SVHN(root='../data',  download=True,split='test', transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)
        if dataset_type=='in':
            fullset=torchvision.datasets.SVHN(root='../data',  download=True,split='test', transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:6])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out':
            fullset=torchvision.datasets.SVHN(root='../data',  download=True,split='test', transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[6:])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            # dataset=get_split_set(fullset,args,id_list[6:])
            # print(id_list[6:])
            return torch.utils.data.DataLoader(fullset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    if args.in_dataset=='cifar4':
        args.class_num=4
        id_list=[0,1,8,9]
        mean,std=[0.4964, 0.5064, 0.5172],[0.2599, 0.2576, 0.2746]     
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
        animal_idx=get_cifar100_animal_label_idx()
        np.random.seed(args.seed)
        np.random.shuffle(animal_idx)

        if dataset_type=='train':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=tr_transform)
            dataset=get_split_set(fullset,args,id_list)
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list)
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)
        if dataset_type=='in':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,id_list)
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out10':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,animal_idx[:10])
            fullset=torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=te_transform)
            dataset1=get_split_set(fullset,args,animal_idx[:10])
            dataset=dataset.__add__(dataset1)
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out50':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            dataset=get_split_set(fullset,args,animal_idx)
            fullset=torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=te_transform)
            dataset1=get_split_set(fullset,args,animal_idx)
            dataset=dataset.__add__(dataset1)
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs) 
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=te_transform)
            val_id_list=[2,3,4,5,6,7]
            dataset=get_split_set(fullset,args,val_id_list)
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    if args.in_dataset=='tiny_imagenet':
        args.class_num=20
        # id_list = open('../data/TinyImagenet/wnids.txt').read().splitlines()
        id_list=list(range(200))
        np.random.seed(args.seed)
        np.random.shuffle(id_list)
        mean,std=cal_mean_std(args,id_list[:20])   
        tr_transform=transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        te_transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])

        if dataset_type=='train':
            fullset=torchvision.datasets.ImageFolder(root='../data/TinyImagenet/train',transform=tr_transform)
            dataset=get_split_set(fullset,args,id_list[:20])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        if dataset_type=='test':
            fullset=torchvision.datasets.ImageFolder(root='../data/TinyImagenet/val/images',transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:20])
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)
        if dataset_type=='in':
            fullset=torchvision.datasets.ImageFolder(root='../data/TinyImagenet/val/images',transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[:20])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='out':
            fullset=torchvision.datasets.ImageFolder(root='../data/TinyImagenet/val/images',transform=te_transform)
            dataset=get_split_set(fullset,args,id_list[20:])
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        if dataset_type=='val':
            fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=te_transform)
            # dataset=get_split_set(fullset,args,id_list[6:])
            # print(id_list[6:])
            return torch.utils.data.DataLoader(fullset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)


def get_split_set(src,args,id_list):
    print(id_list)
    image_path = []
    image_label = []
    if args.in_dataset=='svhn6':
        for i in range(len(src.data)):
            if int(src.labels[i]) in id_list:
                image_path.append(src.data[i])
                image_label.append(id_list.index(src.labels[i]))
        src.data = image_path
        src.labels = image_label
    if args.in_dataset=='cifar6' or args.in_dataset=='cifar4':
        for i in range(len(src.data)):
            if int(src.targets[i]) in id_list:
                image_path.append(src.data[i])
                image_label.append(id_list.index(src.targets[i]))
        src.data = image_path
        src.targets = image_label
    if args.in_dataset=='tiny_imagenet':
        split_samples=[]
        for i in range(len(src.samples)):
            if int(src.samples[i][1]) in id_list:
                smp=tuple((src.samples[i][0],id_list.index(src.samples[i][1])))
                split_samples.append(smp)
        src.samples = split_samples
    return src


def cal_mean_std(args,id_list):
    if os.path.exists('{}/mean_std.pt'.format(args.save)):
        mean_std=torch.load('{}/mean_std.pt'.format(args.save))
        return mean_std['mean'],mean_std['std']
    if args.in_dataset=='cifar6':
        fullset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    if args.in_dataset=='svhn6':
        fullset = torchvision.datasets.SVHN(root='../data',  download=True,split='train', transform=transforms.Compose([transforms.ToTensor()]))
    if args.in_dataset=='tiny_imagenet':
        fullset=torchvision.datasets.ImageFolder('../data/TinyImagenet/train', transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
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

def get_cifar100_animal_label_idx():
    fullset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,\
    transform=transforms.Compose([transforms.ToTensor()]))
    animal_label=['beaver','dolphin','otter','seal','whale',
        'aquarium_fish','flatfish','ray','shark','trout',
        'bee','beetle','butterfly','caterpillar','cockroach',
        'bear','leopard','lion','tiger','wolf',
        'camel','cattle','chimpanzee','elephant','kangaroo',
        'fox','porcupine','possum','raccoon','skunk',
        'crab','lobster','snail','spider','worm',
        'baby','girl','man','woman','boy',
        'crocodile','dinosaur','lizard','snake','turtle',
        'hamster','mouse','rabbit','shrew','squirrel'
        ]
    sort_label=sorted(animal_label)
    animal_index=[]
    for i in sort_label:
        animal_index.append(fullset.class_to_idx[i]) 
    return animal_index