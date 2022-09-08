from configs import *
from utils import *
import os
from cal_metric import *
import shutil
from get_data import *


def ensemble(args,model_list):
    if os.path.exists('{}/score'.format(ensemble_path)):
        shutil.rmtree('{}/score'.format(ensemble_path))
    os.makedirs('{}/score'.format(ensemble_path))

    dataset_list=[]
    dataset_list.append(args.in_dataset)
    if args.in_dataset=='cifar4':
        dataset_list.append('{}_out10'.format(args.in_dataset))
        dataset_list.append('{}_out50'.format(args.in_dataset))
    else:
        dataset_list.append('{}_out'.format(args.in_dataset))
    dataset_list.append('{}_val'.format(args.in_dataset))

    score_type_list=['Pred','Pred_Tsc']
    for ds in dataset_list:
        for st in score_type_list:
            for i in range (len(model_list)):
                each_score = torch.load('{}/score/{}_{}.pt'.format(model_list[i],ds,st))
                each_score = torch.unsqueeze(each_score,dim=0)
                if i==0:
                    total_score = each_score
                else:
                    total_score = torch.cat((total_score,each_score),dim=0)
            mean_score=torch.mean(total_score,dim=0)
            en_score = torch.sum(mean_score.mul(torch.log(mean_score)),dim=1)
            for i in range (len(model_list)):
                each_bias_score=torch.sum(-mean_score.mul(torch.log(total_score[i])),dim=1)
                if i==0:
                    total_bias_score=each_bias_score
                else:
                    total_bias_score=torch.add(total_bias_score,each_bias_score)
            bias_score=total_bias_score/len(model_list)
            
            calibra_score = (1-args.calibration)*en_score - args.calibration * bias_score
            torch.save(mean_score,'{}/score/{}_{}.pt'.format(ensemble_path,ds,st))
            torch.save(bias_score,'{}/score/{}_bias_{}.pt'.format(ensemble_path,ds,st))
            torch.save(calibra_score,'{}/score/{}_cali_{}.pt'.format(ensemble_path,ds,st))

    args.save=ensemble_path
    info=dict()
    info['magnitude']=0
    torch.save(info,"{}/info_{}.pt".format(ensemble_path,args.calibration))

def get_acc(args,model_list):
    in_dataloader=get_dataloader(args,'in')
    nTotal = len(in_dataloader.dataset)
    total_target=torch.empty(0,dtype=int).cuda()
    for data, target in in_dataloader:
        target = target.cuda()
        total_target=torch.cat((total_target,target),dim=0)

    accs=[]
    for i in range (len(model_list)):
        each_score = torch.load('{}/score/{}_Pred.pt'.format(model_list[i],args.in_dataset))
        Pred_each=torch.max(each_score,dim=1)[1]
        incorrect = Pred_each.ne(total_target.data).cpu().sum().item()
        accs.append(round(100-100.*incorrect/nTotal,2))
        if i==0:
            total_score = each_score
        else:
            total_score = total_score+each_score
    Pred=torch.max(total_score,dim=1)[1]

    incorrect = Pred.ne(total_target.data).cpu().sum().item()
    acc_ood=100-100.*incorrect/nTotal
    with open('{}/acc_{}_{}.txt'.format(args.save,args.infer,args.ood_batch_size),'wt') as f:
        print('acc:{:.2f}'.format(acc_ood),file=f)
        print('{},accs:'.format(args.save),file=f)
        print(accs,file=f)




if __name__=='__main__':
    args=Config().data
    set_seed(args.seed)
    if args.in_dataset=='tiny_imagenet':
        args.class_num=20
    elif args.in_dataset=='cifar4':
        args.class_num=4
    else:
        args.class_num=6
    model_list=get_ensemble_list(args)
    print(model_list)
    if args.train_type=='LSR':
        ensemble_path = '../models/{}_seed{}_{}/{}_esbl{}_range{}/ensemble_epochs{}'.format(
            args.net,args.seed,args.in_dataset,
            args.train_type,args.ensemble_num,args.smooth_range,args.epochs)
    if args.train_type=='ILSR':
        ensemble_path = '../models/{}_seed{}_{}/{}_esbl{}_range{}_penalty{}/ensemble_epochs{}'.format(
            args.net,args.seed,args.in_dataset,
            args.train_type,args.ensemble_num,args.smooth_range,args.penalty,args.epochs)
    args.save=ensemble_path

    count=int(1/args.calibration_interval)
    for i in range(count+1):
        args.calibration=round(i*args.calibration_interval,2)
        if 'cal_score' in args.excute_list:
            ensemble(args,model_list)
        if 'cal_metric' in args.excute_list:
            cal_metric(args,['tnr95','auroc'])
        if 'print_info' in args.excute_list:
            print_info(args)
    if 'plot_fig' in args.excute_list:
        plt_fig(args)
    if 'write_to_excel' in args.excute_list:
        write_to_excel(args)
    if 'cal_acc' in args.excute_list:
        get_acc(args,model_list)