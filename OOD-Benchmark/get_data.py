import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

transform_c10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_c100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])
transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])
transforms_list=dict()
transforms_list['cifar10']=transform_c10
transforms_list['cifar100']=transform_c100

trainTransform_c10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
trainTransform_c100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])
trainTransform_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])
TrainTransform_list=dict()
TrainTransform_list['cifar10']=trainTransform_c10
TrainTransform_list['cifar100']=trainTransform_c100


def add_symmetric_noise(args,dataset):
    np.random.seed(0)
    noisy_label=np.zeros_like(dataset.targets,dtype=np.int64)
    labels=np.asarray(dataset.targets)
    for i in range(args.class_num):
        indices = np.where(labels == i)[0]
        np.random.seed(0)
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < args.noise_ratio * len(indices):
                new_label = np.random.randint(args.class_num, dtype=np.int64)
                while(new_label==labels[idx]):#To exclude the original label
                    new_label = np.random.randint(args.class_num, dtype=np.int64)
                noisy_label[idx] = new_label
            else:
                noisy_label[idx] = labels[idx]
    dataset.targets=noisy_label
    return dataset

def add_asymmetric_noise(args,dataset):
    noisy_label=np.zeros_like(dataset.targets,dtype=np.int64)
    if args.in_dataset=='cifar10':
        noise_label_list=np.asarray([0,1,0,5,7,3,6,7,8,1],dtype=np.int64)
    if args.in_dataset=='cifar100':
        noise_label_list=np.asarray([
            51,32,11,42,30,20,7,14,13,10,16,35,17,48,18,19,28,37,24,21,
            25,31,39,33,6,84,45,29,61,44,55,38,67,49,63,46,50,68,15,40,
            86,69,43,88,78,77,98,52,58,60,65,53,56,57,62,72,59,83,90,96,
            71,9,70,64,66,74,75,73,76,81,82,23,95,91,80,34,12,79,93,99,
            36,85,92,0,94,89,87,22,97,41,8,1,54,27,5,4,47,3,2,26
        ],dtype=np.int64)
    labels=np.asarray(dataset.targets)
    for i in range(args.class_num):
        indices = np.where(labels == i)[0]
        np.random.seed(0)
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < args.noise_ratio * len(indices):
                noisy_label[idx] = noise_label_list[i]
            else:
                noisy_label[idx] = labels[idx]
    dataset.targets=noisy_label
    return dataset

def get_dataloader(args,dataset_type):
    
    # args.num_workers=args.num_workers*torch.cuda.device_count()
    args.train_batch_size = args.train_batch_size*torch.cuda.device_count()
    # args.ood_batch_size = args.ood_batch_size*torch.cuda.device_count()
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}

    if args.in_dataset=='cifar10'and dataset_type=='train':
        args.class_num=10
        dataset=torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=TrainTransform_list[args.in_dataset])
        if args.noise_ratio==0:
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        elif args.noise_type=='sym':
            noise_dataset=add_symmetric_noise(args,dataset)
            return torch.utils.data.DataLoader(noise_dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        else:
            noise_dataset=add_asymmetric_noise(args,dataset)
            return torch.utils.data.DataLoader(noise_dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
    if args.in_dataset=='cifar10'and dataset_type=='train_thred':
        args.class_num=10
        dataset=torchvision.datasets.CIFAR10(root='../data', train=True, download=True,transform=TrainTransform_list[args.in_dataset])
        if args.noise_ratio==0:
            return torch.utils.data.DataLoader(dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        elif args.noise_type=='sym':
            noise_dataset=add_symmetric_noise(args,dataset)
            return torch.utils.data.DataLoader(noise_dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
        else:
            noise_dataset=add_asymmetric_noise(args,dataset)
            return torch.utils.data.DataLoader(noise_dataset,batch_size=args.ood_batch_size, shuffle=False, **kwargs)
    if args.in_dataset=='cifar10'and dataset_type=='test':
        args.class_num=10
        dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_list[args.in_dataset])
        return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)

    if args.in_dataset=='cifar10'and dataset_type=='in':
        args.class_num=10
        dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_list[args.in_dataset])
        return torch.utils.data.DataLoader(dataset, batch_size=args.ood_batch_size, shuffle=False)


    if args.in_dataset=='cifar100'and dataset_type=='train':
        args.class_num=100
        dataset=torchvision.datasets.CIFAR100(root='../data', train=True, download=True,transform=TrainTransform_list[args.in_dataset])
        if args.noise_ratio==0:
            return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        elif args.noise_type=='sym':
            noise_dataset=add_symmetric_noise(args,dataset)
            return torch.utils.data.DataLoader(noise_dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)
        else:
            noise_dataset=add_asymmetric_noise(args,dataset)
            return torch.utils.data.DataLoader(noise_dataset,batch_size=args.train_batch_size, shuffle=True, **kwargs)

    if args.in_dataset=='cifar100'and dataset_type=='test':
        args.class_num=100
        dataset=torchvision.datasets.CIFAR100(root='../data', train=False, download=True,transform=transforms_list[args.in_dataset])
        return torch.utils.data.DataLoader(dataset,batch_size=args.train_batch_size, shuffle=False, **kwargs)

    if args.in_dataset=='cifar100'and dataset_type=='in':
        args.class_num=100
        dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transforms_list[args.in_dataset])
        return torch.utils.data.DataLoader(dataset, batch_size=args.ood_batch_size, shuffle=False)
