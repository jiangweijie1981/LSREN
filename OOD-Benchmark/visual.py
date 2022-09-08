
#可视化id数据集的均值和偏差
from numpy.core.fromnumeric import mean
import torch

pt_path='../models/dense_seed1_cifar100_sym0.0/LSR_esbl25_range0.5/ensemble_epochs100'
id,od,nd,st='cifar100_out','Imagenet_resize','','Pred_Tsc'

alpha=0.3

id_mean=torch.load('{}/score/{}_{}.pt'.format(pt_path,id,st))
id_mean=-torch.sum(id_mean*torch.log(id_mean),dim=1)
id_mean=id_mean-torch.mean(id_mean).item()
print(torch.std_mean(id_mean))

id_bias=torch.load('{}/score/{}_bias_{}.pt'.format(pt_path,id,st))
# bias_score=bias_score-torch.mean(bias_score).item()
print(torch.std_mean(id_bias))
id_cali=(1-alpha)*id_mean+alpha*id_bias

od_mean=torch.load('{}/score/{}_{}.pt'.format(pt_path,id,st))
od_mean=-torch.sum(od_mean*torch.log(od_mean),dim=1)
od_mean=od_mean-torch.mean(od_mean).item()
od_bias=torch.load('{}/score/{}_bias_{}.pt'.format(pt_path,od,st))
od_cali=(1-alpha)*od_mean+alpha*od_bias

# max_std=100
# for i in range (21):
#     alpha=i*0.05
#     id_cali=(1-alpha)*id_mean+alpha*id_bias
#     std=torch.mean(id_cali).item()
#     if std<max_std:
#         max_std=std
#         print('alpha:{}, std:{}'.format(alpha,std))




plt_data=[id_mean,id_cali]
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,5))

color_brief=['r','b','g','y','c','m','w']

for i in range(len(plt_data)):
    # plt.hist(plt_data[i].detach().cpu().numpy(), bins=1000,range=(min_value,max_value), density=False, histtype='bar',\
    #     color=color_brief[i],alpha=1-0.3*i,log=True)
    plt.hist(plt_data[i].detach().cpu().numpy(), bins=1000, density=False, histtype='bar',\
        color=color_brief[i],alpha=1-0.3*i,log=True)
# plt.show()
