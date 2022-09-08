import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from configs import *
from get_data import *
from utils import *
import torch.nn.functional as F

def eva_metric(msp_in,msp_out):
    tnr95 = 1-torch.sum(torch.sum(msp_out>msp_in[int(round(len(msp_in)*0.05))]))/float(len(msp_out))
    thred = torch.sort(torch.cat((msp_in,msp_out),dim=0))[0]
    auroc = 0.0
    fprTemp = 1.0
    for thred_idx in range (len(thred)):
        tpr = torch.sum(torch.sum(msp_in >= thred[thred_idx])) / float(len(msp_in))
        fpr = torch.sum(torch.sum(msp_out > thred[thred_idx])) / float(len(msp_out))
        auroc += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    auroc += fpr * tpr
    return tnr95.item(),auroc.item()

def evaluate(args):
    print(args.save)
    dataLoader=get_dataloader(args,'in')
    net=load_net(args,'{}/latest-{}.pth'.format(args.save,args.epochs))
    msp_in=torch.empty(0).cuda()

    with torch.no_grad():
        net.eval()
        for batch_idx, (data, target) in enumerate(dataLoader):
            data, target = data.cuda(), target.cuda()
            outputs = net(data)[2]
            # softmax_output=F.softmax(outputs,dim=1)
            Msp_batch = torch.max(outputs,dim=1)[0]
            msp_in=torch.cat((msp_in,Msp_batch),dim=0)
    msp_in = torch.sort(msp_in)[0]
    print('cal msp_in end.')
    msp_out=torch.empty(0).cuda()
    dataLoader=get_dataloader(args,'out')
    with torch.no_grad():
        net.eval()
        for batch_idx, (data, target) in enumerate(dataLoader):
            data, target = data.cuda(), target.cuda()
            outputs = net(data)[2]
            # softmax_output=F.softmax(outputs,dim=1)
            Msp_batch = torch.max(outputs,dim=1)[0]
            msp_out=torch.cat((msp_out,Msp_batch),dim=0)
    print('cal msp_out end.')
    tnr95_1,auroc_1=eva_metric(msp_in,msp_out)
    print_info='tnr95_1: {:.4f}, auroc_1: {:.4f}'.format(tnr95_1,auroc_1)
    print('cal ood1 end.')

    mean_std=torch.load('{}/mean_std.pt'.format(args.save))
    te_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std['mean'],mean_std['std'])
        ])

    out_datasets=['Imagenet','Imagenet_resize','LSUN','LSUN_resize','iSUN']
    tnr95_2,auroc_2=0,0
    for item in out_datasets:
        msp_out=torch.empty(0).cuda()
        out_dataset = torchvision.datasets.ImageFolder("../data/{}".format(item), transform=te_transform)
        dataLoader = torch.utils.data.DataLoader(out_dataset, batch_size=args.ood_batch_size, shuffle=True)
        with torch.no_grad():
            net.eval()
            for batch_idx, (data, target) in enumerate(dataLoader):
                data, target = data.cuda(), target.cuda()
                outputs = net(data)[2]
                # softmax_output=F.softmax(outputs,dim=1)
                Msp_batch = torch.max(outputs,dim=1)[0]
                msp_out=torch.cat((msp_out,Msp_batch),dim=0)
        print('cal msp_{} end.'.format(item))
        tnr95,auroc=eva_metric(msp_in,msp_out)
        print_info+='\n{}, tnr95: {:.4f}, auroc: {:.4f}'.format(item,tnr95,auroc)
        tnr95_2+=tnr95
        auroc_2+=auroc
    print('cal ood2 end.')
    print_info+='\ntnr95_2: {:.4f}, auroc_2: {:.4f}'.format(tnr95_2/len(out_datasets),auroc_2/len(out_datasets))

    with open('{}/evaluation.txt'.format(args.save),'wt') as f:
        print(print_info,file=f)
    eva=dict()
    eva['auroc1'],eva['auroc2']=auroc_1,auroc_2/len(out_datasets)
    torch.save(eva,'{}/eva.pt'.format(args.save))


if __name__=='__main__':
    args=Config().data
    evaluate(args)