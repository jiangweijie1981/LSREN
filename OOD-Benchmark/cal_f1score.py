from cgi import print_arguments
from numpy.lib.function_base import average
import sklearn
from configs import Config
import torch
from sklearn import metrics
from get_data import *
from configs import *
from utils import *
import torch.nn.functional as F
import time
import os

args=Config().data
args.in_dataset='cifar10'
args.num_workers=0
tr_dataloader=get_dataloader(args,'train_thred')
thred_coef=args.f1_thred_coef
# calibration=args.calibration


score_path='../models/{}_seed{}_cifar10_sym0.0/LSR_esbl25_range0.5/ensemble_epochs100/score'.format(args.net,args.seed)
t0 = time.time()
if not os.path.exists('{}/train_mean_{}.pt'.format(score_path,args.infer)) or not os.path.exists('{}/train_bias_{}.pt'.format(score_path,args.infer)) :
    model_list=get_ensemble_list(args)
    for i in range (len(model_list)):
        model_path="{}/latest-100.pth".format(model_list[i])
        net=load_net(args,model_path)
        with torch.no_grad():
            if args.infer=='bn':
                net.eval()
            Pred_Tsc=torch.empty(0,dtype=torch.float64).cuda()
            for batch_idx, data in enumerate(tr_dataloader):
                images, _ = data
                inputs = images.cuda()
                outputs = net(inputs)[2]
                # Using temperature scaling
                outputs_t = outputs / args.temperature
                softmax_output_t=F.softmax(outputs_t,dim=1).to(torch.float64)
                Pred_Tsc=torch.cat((Pred_Tsc,softmax_output_t),dim=0)
        Pred_Tsc=torch.unsqueeze(Pred_Tsc,dim=0)
        if i==0:
            total_Pred_Tsc = Pred_Tsc
        else:
            total_Pred_Tsc = torch.cat((total_Pred_Tsc,Pred_Tsc),dim=0)
        print('cal {}/{} train models\n'.format(i,len(model_list)))
    mean_score=torch.mean(total_Pred_Tsc,dim=0)
    for i in range (len(model_list)):
        each_bias_score=torch.sum(-mean_score.mul(torch.log(total_Pred_Tsc[i])),dim=1)
        if i==0:
            total_bias_score=each_bias_score
        else:
            total_bias_score=torch.add(total_bias_score,each_bias_score)
    bias_score=total_bias_score/len(model_list)
    torch.save(mean_score,'{}/train_mean_{}.pt'.format(score_path,args.infer))
    torch.save(bias_score,'{}/train_bias_{}.pt'.format(score_path,args.infer))
mean_score=torch.load('{}/train_mean_{}.pt'.format(score_path,args.infer))
bias_score=torch.load('{}/train_bias_{}.pt'.format(score_path,args.infer))

txt_path='../models/{}_seed{}_cifar10_sym0.0/LSR_esbl25_range0.5/ensemble_epochs100'.format(args.net,args.seed)
cali_list=[round(i*0.05,2) for i in range(21)]
score_type_list=['Pred','Pred_Tsc']
ood_list=['Imagenet','Imagenet_resize','LSUN','LSUN_resize']
score_total=dict()

for calibration in cali_list:
    print(calibration)
    score_total[calibration]=dict()
    en_score = torch.sum(mean_score.mul(torch.log(mean_score)),dim=1)
    calibra_score = (1-calibration)*en_score - calibration * bias_score
    #获取阈值
    sort_score = torch.sort(calibra_score)[0]
    thred=sort_score[int(round(len(sort_score)*(1-thred_coef)))].item()

    for st in score_type_list:
        print(st)
        score_total[calibration][st]=[]
        #获取id分数
        id_pred_score = torch.load('{}/cifar10_out_{}.pt'.format(score_path,st))
        id_pred=id_pred_score.data.max(1)[1]
        id_en_score= torch.sum(id_pred_score.mul(torch.log(id_pred_score)),dim=1)
        id_bias_score=torch.load('{}/cifar10_out_bias_{}.pt'.format(score_path,st))
        id_cali_score=(1-calibration)*id_en_score - calibration * id_bias_score
        # thred_test=torch.sort(id_cali_score)[0][int(round(len(id_cali_score)*(1-0.95)))].item()
        ood_score_list=[]
        total=0
        for ood in ood_list:
            #获取ood分数
            ood_pred_score = torch.load('{}/{}_{}.pt'.format(score_path,ood,st))
            ood_pred=ood_pred_score.data.max(1)[1]
            ood_en_score= torch.sum(ood_pred_score.mul(torch.log(ood_pred_score)),dim=1)
            ood_bias_score=torch.load('{}/{}_bias_{}.pt'.format(score_path,ood,st))
            ood_cali_score=(1-calibration)*ood_en_score - calibration * ood_bias_score
            # tnr=1-torch.sum(torch.sum(ood_cali_score>thred))/float(len(ood_cali_score))
            # print('tnr:{}'.format(tnr.item()))
            #预测标签
            pred_old=torch.cat((id_pred,ood_pred),dim=0)
            ood_all=torch.cat((id_cali_score,ood_cali_score),dim=0)
            pred_temp=torch.ones_like(pred_old)*11
            pred_all=torch.where(ood_all<=thred,pred_temp,pred_old)
            # print('thred:{}, id_same_num:{}, ood_same_num:{}'.format(thred-thred_test,torch.sum(id_cali_score==thred),torch.sum(ood_cali_score==thred)))
            #获取gt
            gt_id=torch.empty(0,dtype=int).cuda()
            in_dataloader=get_dataloader(args,'in')
            with torch.no_grad():
                for __, target in in_dataloader:
                    target = target.cuda()
                    gt_id=torch.cat((gt_id,target),dim=0)
            gt_ood=torch.ones_like(ood_pred)*11
            gt_all=torch.cat((gt_id,gt_ood),dim=0)
            #计算F1分数
            score=sklearn.metrics.f1_score(gt_all.cpu(),pred_all.cpu(),average='macro')
            total+=score
            ood_score_list.append('{:.4f}   '.format(score))
        ood_score_list.append('{:.4f}   '.format(total/4))
        score_total[calibration][st].append(ood_score_list)
        
with open('{}/f1_socre_test_{}.txt'.format(txt_path,args.infer),'wt') as f:
    for st in score_type_list:
        for calibration in cali_list:        
            print('type:{},  cali:{}'.format(st,calibration),file=f)
            print(score_total[calibration][st],file=f)
            print('\n',file=f)
                
        
