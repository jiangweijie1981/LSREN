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
from torch import float64
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from get_data import get_dataloader
import time
import math
from pylab import *

def get_visual_data(args,seed,smoothness):
    set_dataset_path(args)
    dataset_config(args)
    args.output_path = '{}/{}'.format(args.dataset_config_path,args.net)
    file_path='{}/models/seed{}_epochs{}_s{:.6f}_proportion{}'.format(args.output_path,seed,args.epochs,smoothness,args.proportion)
    
    z,labels=torch.empty(0),torch.empty(0)
    net=load_net(args)
    net.eval()
    count=0
    inferLoader=get_dataloader(args,'infer')
    for data, target in inferLoader:
        data, target = data.cuda(), target.cuda()
        fea = net(data)[3].detach()
        if count==0:
            z=fea
            labels=target
        else:
            z=torch.cat((z,fea),dim=0)
            labels=torch.cat((labels,target),dim=0)
        count+=1
    noiseLoader=get_dataloader(args,'noise')
    for data,__ in noiseLoader:
        data = data.cuda()
        fea = net(data)[3].detach()
        z=torch.cat((z,fea),dim=0)
        labels=torch.cat((labels,torch.ones((data.size(0)),dtype=int).cuda()*10),dim=0)
    print('Inferance end, samples/label: {}/{}'.format(len(z),len(labels)))

    sample_length=800
    t0=time.time()
    sample_z=torch.empty((sample_length*11,z.size(1)))
    count=[0]*11
    for i in range(z.size(0)):
        if (count[labels[i]]<sample_length):
            sample_z[labels[i]*sample_length+count[labels[i]]]=z[i]
            count[labels[i]]+=1
    print('Sample end, time:{:.1f}'.format(time.time()-t0))
    
    t0=time.time()
    if not os.path.exists('{}_tsne.pt'.format(file_path)):
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_z=tsne.fit_transform(sample_z)
        torch.save(tsne_z,'{}_tsne.pt'.format(file_path))
    visual_z=torch.load('{}_tsne.pt'.format(file_path))
    print('Transform {}, time:{:.1f}'.format(len(visual_z),time.time()-t0))
    return visual_z,file_path,sample_length

def visual(args):
    colors=['darkcyan','#85C085','y','k','orange','#FCCCAD','#F9A975','#F7863C','pink','darkorchid','#4D85BD',\
    'dodgerblue','limegreen','thistle','lightblue','gold','brown','deeppink','maroon','pink','sienna']
    scatter_size=1
    font_size=8
    fig = plt.figure(figsize=(7.5,3.5),dpi=300)
    subplots_adjust(left=0.05,right=0.87)


    ax = plt.subplot(241)
    plt.title('(a) Smoothness=0.0',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,0,0)
    visual_list=args.visual_class_list
    fig_handles=[]
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(visual_z[k*sample_length:(k+1)*sample_length, 0]*math.cos(math.pi/4)+visual_z[k*sample_length:(k+1)*sample_length, 1]*math.sin(math.pi/4),\
            -visual_z[k*sample_length:(k+1)*sample_length, 0]*math.sin(math.pi/4)+visual_z[k*sample_length:(k+1)*sample_length, 1]*math.cos(math.pi/4),\
            s=scatter_size, c=colors[k], linewidths=0)
    ax.set_ylabel(ylabel='Different initializations',fontsize=font_size,labelpad=10)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]


    ax = plt.subplot(242)
    plt.title('(b) Smoothness=0.0',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,5,0)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(visual_z[k*sample_length:(k+1)*sample_length, 1],visual_z[k*sample_length:(k+1)*sample_length, 0],\
                s=scatter_size, c=colors[k], linewidths=0)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    ax = plt.subplot(243)
    plt.title('(c) Smoothness=0.0',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,6,0)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(-visual_z[k*sample_length:(k+1)*sample_length, 1],visual_z[k*sample_length:(k+1)*sample_length, 0],\
            s=scatter_size, c=colors[k], linewidths=0)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    ax = plt.subplot(244)
    plt.title('(d) Smoothness=0.0',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,7,0)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        rotate=math.pi*3/4
        ax.scatter(-visual_z[k*sample_length:(k+1)*sample_length, 0]*math.cos(rotate)-visual_z[k*sample_length:(k+1)*sample_length, 1]*math.sin(rotate),\
            -visual_z[k*sample_length:(k+1)*sample_length, 0]*math.sin(rotate)+visual_z[k*sample_length:(k+1)*sample_length, 1]*math.cos(rotate),\
            s=scatter_size, c=colors[k], linewidths=0)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    ax = plt.subplot(245)
    plt.title('(e) Smoothness=0.1',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,0,0.1)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(-visual_z[k*sample_length:(k+1)*sample_length, 0],visual_z[k*sample_length:(k+1)*sample_length, 1],\
            s=scatter_size, c=colors[k], linewidths=0)
    ax.set_ylabel(ylabel='Identical initializations',fontsize=font_size,labelpad=10)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    ax = plt.subplot(246)
    plt.title('(f) Smoothness=0.2',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,0,0.2)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(visual_z[k*sample_length:(k+1)*sample_length, 0],visual_z[k*sample_length:(k+1)*sample_length, 1],\
            s=scatter_size, c=colors[k], linewidths=0)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    ax = plt.subplot(247)
    plt.title('(g) Smoothness=0.4',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,0,0.4)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(-visual_z[k*sample_length:(k+1)*sample_length, 0]*math.cos(-math.pi/4)-visual_z[k*sample_length:(k+1)*sample_length, 1]*math.sin(-math.pi/4),\
            -visual_z[k*sample_length:(k+1)*sample_length, 0]*math.sin(-math.pi/4)+visual_z[k*sample_length:(k+1)*sample_length, 1]*math.cos(-math.pi/4),\
            s=scatter_size, c=colors[k], linewidths=0)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    ax = plt.subplot(248)
    plt.title('(h) Smoothness=0.6',fontsize=font_size)
    visual_z,file_path,sample_length=get_visual_data(args,0,0.6)
    visual_list=args.visual_class_list
    for i in range(len(visual_list)):
        k=visual_list[i]
        ax.scatter(visual_z[k*sample_length:(k+1)*sample_length, 0],visual_z[k*sample_length:(k+1)*sample_length, 1],\
            s=scatter_size, c=colors[k], linewidths=0)
    plt.xticks([])
    plt.yticks([])
    [ax.spines[item].set_linewidth(0) for item in ['top','bottom','left','right']]

    recs=[]
    for i in args.visual_class_list:
        recs.append(mpatches.Rectangle((0,0),0.2,0.2,fc=colors[i]))
    labels=['ID class 1','ID class 2','ID class 3  ','SOOD data  ','OOD data']
    fig.legend(recs,labels,fontsize=font_size,frameon=False,handletextpad=0.2,handlelength=0.8,bbox_to_anchor=(1.0,0.95))
     
    plt.savefig('{}.png'.format(file_path))
    plt.savefig('{}.eps'.format(file_path),format='eps')
    plt.close()

if __name__=='__main__':
    args=Config().data
    args.train_class_list=[5,6,7]
    args.visual_class_list=[5,6,7,1,10]
    visual(args)