from torch.serialization import load
import torchvision
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import torch
import numpy as np
from configs import *
from get_data import *
from utils import *
import torch.nn.functional as F
import matplotlib.patches as mpatches

def plot_embedding(z, labels, path_prefix,title,args):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import matplotlib as mpl
    mpl.use('Agg')
    font_size=3.0
    labels=labels.cpu().numpy()
    colors=['k','gray','r','b','y','coral','orange','teal','darkorchid','darkcyan','dodgerblue','limegreen','thistle','lightblue',
    'gold','g','brown','deeppink','maroon','pink','sienna']
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    if not os.path.exists('{}/tsne_z.pt'.format(args.save)):
        data = tsne.fit_transform(z.cpu())
        torch.save(data,'{}/tsne_z.pt'.format(args.save))
    else:
        data=torch.load('{}/tsne_z.pt'.format(args.save))
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # plt.style.use(['science','ieee'])
    plt.figure(dpi=600,figsize=[1.75,1.5])
    

    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=colors[labels[i]],s=0.2,linewidths=0.1)

    if args.in_dataset=='cifar6':
        classes = [' OOD Sample  ',' SOOD Sample  ','','','','','',' ID Sample']
    else:
        classes = [' OOD Sample  ',' SOOD Sample  ',' ID Sample']
    recs = []
    for i in range(0,len(classes)):
        recs.append(mpatches.Rectangle((0,0),0.2,0.2,fc=colors[i]))
    plt.legend(recs,classes,fontsize=font_size,frameon=False,loc=4,ncol=8,handletextpad=0.2,\
        columnspacing=0.2,handlelength=0.8,bbox_to_anchor=(1.0,-0.1))

    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    # plt.rcParams['savefig.dpi'] = 300
    ax = plt.gca()
    ax.spines['top'].set_linewidth(0.2)
    ax.spines['bottom'].set_linewidth(0.2)
    ax.spines['left'].set_linewidth(0.2)
    ax.spines['right'].set_linewidth(0.2)

    # plt.axis('off')

    plt.savefig('visual/fig/{}.png'.format(title))
    plt.savefig('visual/fig/{}.eps'.format(title),format='eps')
    plt.close()


args=Config().data
args.num_workers=0
# args.in_dataset='cifar6_100'
# args.class_num=6
# args.net='resnet_inn'
# args.epochs=200

args.save = '../models/{}_seed{}_{}_0-{}/LSR_esbl1_range0.5/epochs{}_s0.000000'.format(
    args.net,args.seed,args.in_dataset,args.samples_pre_class_num,args.epochs)

print(args.save)
if args.in_dataset=='cifar6':
    args.class_num=6
else:
    args.class_num=80

if not os.path.exists('{}/visual_fea_z.pt'.format(args.save)):
    args.train_batch_size=100
    dataLoader=get_dataloader(args,'in')
    net=load_net(args,'{}/latest-{}.pth'.format(args.save,args.epochs))
    z,labels,labels1=torch.empty(0).cuda(),torch.empty(0,dtype=int).cuda(),torch.empty(0,dtype=int).cuda()
    args.show_num=10

    with torch.no_grad():
        net.eval()
        for batch_idx, (data, target) in enumerate(dataLoader):
            if batch_idx==args.show_num:
                break
            data, target = data.cuda(), target.cuda()
            outputs = net(data)[3]
            z=torch.cat((z,outputs),dim=0)
            labels=torch.cat((labels,target))
            target1=torch.zeros_like(target)
            labels1=torch.cat((labels1,target1))

    labels=labels+2
    labels1=labels1+2

    args.train_batch_size=100
    dataLoader=get_dataloader(args,'out')
    with torch.no_grad():
        net.eval()
        for batch_idx, (data, target) in enumerate(dataLoader):
            if batch_idx==args.show_num:
                break
            data, target = data.cuda(), target.cuda()
            outputs = net(data)[3]
            z=torch.cat((z,outputs),dim=0)
            target_out=torch.ones_like(target)
            labels=torch.cat((labels,target_out))
            labels1=torch.cat((labels1,target_out))

    mean_std=torch.load('{}/mean_std.pt'.format(args.save))
    te_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std['mean'],mean_std['std'])
        ])

    out_datasets=['Imagenet','Imagenet_resize','LSUN','LSUN_resize','iSUN']

    for item in out_datasets:
        item='Imagenet_resize'
        out_dataset = torchvision.datasets.ImageFolder("../data/{}".format(item), transform=te_transform)
        dataLoader = torch.utils.data.DataLoader(out_dataset, batch_size=args.ood_batch_size, shuffle=True)
        with torch.no_grad():
            net.eval()
            for batch_idx, (data, target) in enumerate(dataLoader):
                if batch_idx==args.show_num/len(out_datasets):
                    break
                data, target = data.cuda(), target.cuda()
                outputs = net(data)[3]
                z=torch.cat((z,outputs),dim=0)
                target_out=torch.zeros_like(target)
                labels=torch.cat((labels,target_out))
                labels1=torch.cat((labels1,target_out))
    torch.save(z,'{}/visual_fea_z.pt'.format(args.save))
    torch.save(labels,'{}/visual_fea_labels.pt'.format(args.save))
    torch.save(labels1,'{}/visual_fea_labels1.pt'.format(args.save))
else:
    z=torch.load('{}/visual_fea_z.pt'.format(args.save))
    labels=torch.load('{}/visual_fea_labels.pt'.format(args.save))
    labels1=torch.load('{}/visual_fea_labels1.pt'.format(args.save))

if args.in_dataset=='cifar6':
    plot_embedding(z,labels,args.save,'{}_visual_fea'.format(args.in_dataset),args)
else:
    plot_embedding(z,labels1,args.save,'{}_visual_fea'.format(args.in_dataset),args)

