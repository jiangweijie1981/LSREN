import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from configs import *
from get_data import *
from utils import *
import torch.nn.functional as F
from pylab import *
import matplotlib.patches as mpatches

args=Config().data
# args.num_workers=0
# args.class_num=6
# args.net='resnet'
# args.epochs=100
fig_dpi=300
fig_bins=50
ood_id=1
font_size=6
infer_type='eval'
log_type=True
density_type=False

#region get_plt_data
base_model= '../models/{}_seed{}_{}_0-{}/LSR_esbl1_range0.5/epochs{}_s0.000000'.format(
    args.net,args.seed,args.in_dataset,args.samples_pre_class_num,args.epochs)
ensb_model= '../models/{}_seed{}_{}_0-{}/LSR_esbl25_range0.5/ensemble_epochs{}'.format(
    args.net,args.seed,args.in_dataset,args.samples_pre_class_num,args.epochs)
print('base_model: {}'.format(base_model))
print('ensb_model: {}'.format(ensb_model))

ood_list=['Imagenet','Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN', 'SVHN']
ood=ood_list[ood_id]


base_id_score=torch.load('{}/{}_score/{}_Pred.pt'.format(base_model,infer_type,args.in_dataset))
base_id_score=-torch.sum(base_id_score*torch.log(base_id_score),dim=1)
base_sood_score=torch.load('{}/{}_score/{}_out_Pred.pt'.format(base_model,infer_type,args.in_dataset))[:8000]
base_sood_score=-torch.sum(base_sood_score*torch.log(base_sood_score),dim=1)
base_ood_score=torch.load('{}/{}_score/{}_{}_Pred.pt'.format(base_model,infer_type,args.in_dataset,ood))[:8000]
base_ood_score=-torch.sum(base_ood_score*torch.log(base_ood_score),dim=1)
# print(base_id_score.size(0))
# print(base_sood_score.size(0))
# print(base_ood_score.size(0))

ensb_id_score=torch.load('{}/{}_score/{}_Pred.pt'.format(ensb_model,infer_type,args.in_dataset))
ensb_id_score=-torch.sum(ensb_id_score*torch.log(ensb_id_score),dim=1)
ensb_sood_score=torch.load('{}/{}_score/{}_out_Pred.pt'.format(ensb_model,infer_type,args.in_dataset))[:8000]
ensb_sood_score=-torch.sum(ensb_sood_score*torch.log(ensb_sood_score),dim=1)
ensb_ood_score=torch.load('{}/{}_score/{}_{}_Pred.pt'.format(ensb_model,infer_type,args.in_dataset,ood))[:8000]
ensb_ood_score=-torch.sum(ensb_ood_score*torch.log(ensb_ood_score),dim=1)
# print(ensb_id_score.size(0))
# print(ensb_sood_score.size(0))
# print(ensb_ood_score.size(0))

ensb_id_bias=torch.load('{}/{}_score/{}_bias_Pred.pt'.format(ensb_model,infer_type,args.in_dataset))
ensb_sood_bias=torch.load('{}/{}_score/{}_out_bias_Pred.pt'.format(ensb_model,infer_type,args.in_dataset))[:8000]
ensb_ood_bias=torch.load('{}/{}_score/{}_{}_bias_Pred.pt'.format(ensb_model,infer_type,args.in_dataset,ood))[:8000]
# print(ensb_id_bias.size(0))
# print(ensb_sood_bias.size(0))
# print(ensb_ood_bias.size(0))

#endregion

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

color_brief=['#F9A968','#85C085','#4D85BD','r','b','g','y','c','m','w']
labels=['ID data','SOOD data','OOD data']
fig=plt.figure(figsize=(3.5,4.5),dpi=fig_dpi)
subplots_adjust(hspace=0.7)

plt_data=[base_id_score,base_sood_score,base_ood_score]
ax=plt.subplot(311)
tick_params(which='major',width=0.3,length=2,pad=1)

plt.title('(a) Single predicion',fontsize=font_size)
ax.set_xlabel(xlabel='Detection score',fontsize=font_size,labelpad=0.5)
ax.set_ylabel(ylabel='Number of Samples',fontsize=font_size,labelpad=1)
plt.xticks(fontsize=font_size,ticks=[0,1,2,3,4,5])
plt.yticks(fontsize=font_size,ticks=[10,100,1000])

for i in range(len(plt_data)):
    plt.hist(plt_data[i].detach().cpu().numpy(), bins=fig_bins, density=density_type, histtype='step',color=color_brief[i],log=log_type,label=labels[i])
[ax.spines[item].set_linewidth(0) for item in ['top','right']]


plt_data=[ensb_id_score,ensb_sood_score,ensb_ood_score]
ax=plt.subplot(312)
tick_params(which='major',width=0.3,length=2,pad=1)
# subplots_adjust(bottom=0.3,top=0.95)
plt.title('(b) Mean of multiple predicions',fontsize=font_size)
ax.set_xlabel(xlabel='Detection score',fontsize=font_size,labelpad=0.5)
ax.set_ylabel(ylabel='Number of Samples',fontsize=font_size,labelpad=1)
plt.xticks(fontsize=font_size,ticks=[0,1,2,3,4,5])
plt.yticks(fontsize=font_size,ticks=[10,100,1000])

for i in range(len(plt_data)):
    plt.hist(plt_data[i].detach().cpu().numpy(), bins=fig_bins, density=density_type, histtype='step',color=color_brief[i],log=log_type,label=labels[i])
[ax.spines[item].set_linewidth(0) for item in ['top','right']]


plt_data=[ensb_id_bias,ensb_sood_bias,ensb_ood_bias]
ax=plt.subplot(313)
tick_params(which='major',width=0.3,length=2,pad=1)
# subplots_adjust(bottom=0.3,top=0.95)
plt.title('(c) Deviation of multiple predicions',fontsize=font_size)
# plt.xlabel('Detection score',fontsize=font_size)
ax.set_xlabel(xlabel='Detection score',fontsize=font_size,labelpad=0.5)
ax.set_ylabel(ylabel='Number of Samples',fontsize=font_size,labelpad=1)
plt.xticks(fontsize=font_size,ticks=[0,1,2,3,4,5])
plt.yticks(fontsize=font_size,ticks=[10,100,1000])

for i in range(len(plt_data)):
    plt.hist(plt_data[i].detach().cpu().numpy(), bins=fig_bins, density=density_type, histtype='step',color=color_brief[i],log=log_type,label=labels[i])
[ax.spines[item].set_linewidth(0) for item in ['top','right']]


recs=[]
for i in range(3):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=color_brief[i]))
fig.legend(recs,labels,fontsize=font_size, frameon=False, handlelength=0.8, handletextpad=0.2,bbox_to_anchor=(1.0,0.9))


plt.savefig('../fig3-2.png'.format(args.in_dataset))
plt.savefig('../fig3-2.eps'.format(args.in_dataset),format='eps')
# plt.show()

