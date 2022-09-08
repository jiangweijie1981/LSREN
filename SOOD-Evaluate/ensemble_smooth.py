from configs import *
from utils import *
import os
from cal_metric import *
import shutil
from get_data import *


def ensemble(args,model_list,out_datasets):
    if 'eval' in args.inferance:
        if os.path.exists('{}/eval_score'.format(ensemble_path)):
            shutil.rmtree('{}/eval_score'.format(ensemble_path))
        os.makedirs('{}/eval_score'.format(ensemble_path))
        dataset_list=[]
        dataset_list.append(args.in_dataset)
        dataset_list.append('{}_val_in'.format(args.in_dataset))
        for item in out_datasets:
            dataset_list.append('{}_{}'.format(args.in_dataset,item))

        score_type_list=['Pred','Pred_Tsc']
        for ds in dataset_list:
            for st in score_type_list:
                for i in range (len(model_list)):
                    each_score = torch.load('{}/eval_score/{}_{}.pt'.format(model_list[i],ds,st))
                    each_score = torch.unsqueeze(each_score,dim=0)
                    if i==0:
                        total_score = each_score
                    else:
                        total_score = torch.cat((total_score,each_score),dim=0)
                mean_score=torch.mean(total_score,dim=0)
                for i in range (len(model_list)):
                    each_bias_score=torch.sum(-mean_score.mul(torch.log(total_score[i])),dim=1)
                    if i==0:
                        total_bias_score=each_bias_score
                    else:
                        total_bias_score=torch.add(total_bias_score,each_bias_score)
                bias_score=total_bias_score/len(model_list)
                en_score = torch.sum(mean_score.mul(torch.log(mean_score)),dim=1)
                calibra_score = (1-args.calibration)*en_score - args.calibration * bias_score
                torch.save(mean_score,'{}/eval_score/{}_{}.pt'.format(ensemble_path,ds,st))
                torch.save(bias_score,'{}/eval_score/{}_bias_{}.pt'.format(ensemble_path,ds,st))
                torch.save(calibra_score,'{}/eval_score/{}_cali_{}.pt'.format(ensemble_path,ds,st))

        args.save=ensemble_path
        info=dict()
        info['magnitude']=0
        torch.save(info,"{}/eval_info_{}.pt".format(ensemble_path,args.calibration))

    if 'ood' in args.inferance:
        if os.path.exists('{}/ood_score'.format(ensemble_path)):
            shutil.rmtree('{}/ood_score'.format(ensemble_path))
        os.makedirs('{}/ood_score'.format(ensemble_path))

        dataset_list=[]
        dataset_list.append(args.in_dataset)
        dataset_list.append('{}_val_in'.format(args.in_dataset))
        for item in out_datasets:
            dataset_list.append('{}_{}'.format(args.in_dataset,item))

        score_type_list=['Pred','Pred_Tsc']
        for ds in dataset_list:
            for st in score_type_list:
                for i in range (len(model_list)):
                    each_score = torch.load('{}/ood_score/{}_{}.pt'.format(model_list[i],ds,st))
                    each_score = torch.unsqueeze(each_score,dim=0)
                    if i==0:
                        total_score = each_score
                    else:
                        total_score = torch.cat((total_score,each_score),dim=0)
                mean_score=torch.mean(total_score,dim=0)
                for i in range (len(model_list)):
                    each_bias_score=torch.sum(-mean_score.mul(torch.log(total_score[i])),dim=1)
                    if i==0:
                        total_bias_score=each_bias_score
                    else:
                        total_bias_score=torch.add(total_bias_score,each_bias_score)
                bias_score=total_bias_score/len(model_list)
                en_score = torch.sum(mean_score.mul(torch.log(mean_score)),dim=1)
                calibra_score = (1-args.calibration)*en_score - args.calibration * bias_score
                torch.save(mean_score,'{}/ood_score/{}_{}.pt'.format(ensemble_path,ds,st))
                torch.save(bias_score,'{}/ood_score/{}_bias_{}.pt'.format(ensemble_path,ds,st))
                torch.save(calibra_score,'{}/ood_score/{}_cali_{}.pt'.format(ensemble_path,ds,st))

        args.save=ensemble_path
        info=dict()
        info['magnitude']=0
        torch.save(info,"{}/ood_info_{}.pt".format(ensemble_path,args.calibration))

def get_acc(args,model_list):
    in_dataloader=get_dataloader(args,'in')
    nTotal = len(in_dataloader.dataset)
    total_target=torch.empty(0,dtype=int).cuda()
    for __, target in in_dataloader:
        target = target.cuda()
        total_target=torch.cat((total_target,target),dim=0)
    if 'eval' in args.inferance:
        eval_accs=[]
        for i in range (len(model_list)):
            info = torch.load('{}/eval_info.pt'.format(model_list[i]))
            eval_accs.append(round(info['acc'].item(),2))
            each_score = torch.load('{}/eval_score/{}_Pred.pt'.format(model_list[i],args.in_dataset))
            if i==0:
                total_score = each_score
            else:
                total_score = total_score+each_score
        Pred=torch.max(total_score,dim=1)[1]

        incorrect = Pred.ne(total_target.data).cpu().sum().item()
        eval_acc=100-100.*incorrect/nTotal
    if 'ood' in args.inferance:
        ood_accs=[]
        for i in range (len(model_list)):
            info = torch.load('{}/ood_info.pt'.format(model_list[i]))
            print(model_list[i])
            print(info)
            ood_accs.append(round(info['acc'].item(),2))
            each_score = torch.load('{}/ood_score/{}_Pred.pt'.format(model_list[i],args.in_dataset))
            if i==0:
                total_score = each_score
            else:
                total_score = total_score+each_score
        Pred=torch.max(total_score,dim=1)[1]

        incorrect = Pred.ne(total_target.data).cpu().sum().item()
        ood_acc=100-100.*incorrect/nTotal

    with open('{}/acc-{}.txt'.format(args.save,args.ood_batch_size),'wt') as f:
        if 'eval' in args.inferance:
            print('{},eval_acc:{:.2f}'.format(args.save,eval_acc),file=f)
            print('eval_accs',file=f)
            print(eval_accs,file=f)
        if 'ood' in args.inferance:
            print('{}, ood_acc:{:.2f}'.format(args.save, ood_acc),file=f)
            print('ood_accs',file=f)
            print(ood_accs,file=f)



if __name__=='__main__':
    args=Config().data
    args.class_num=6
    model_list=get_ensemble_list(args)
    print(model_list)

    if args.train_type=='LSR':
        ensemble_path = '../models/{}_seed{}_{}_{}-{}/{}_esbl{}_range{}/ensemble_epochs{}'.format(
            args.net,args.seed,args.in_dataset,args.dataset_id,args.samples_pre_class_num,
            args.train_type,args.ensemble_num,args.smooth_range,args.epochs)

    args.save=ensemble_path

    out_datasets=['val']
    count=int(1/args.calibration_interval)

    for i in range(count+1):
        args.calibration=round(i*args.calibration_interval,2)
        if 'cal_score' in args.excute_list:
            ensemble(args,model_list,out_datasets)
        if 'cal_metric' in args.excute_list:
            cal_metric(args,['tnr95','auroc'],out_datasets)

    calibration_dict=dict()
    max_auroc,max_calibration=0.0,0.0
    if 'eval' in args.inferance:
        for i in range(count+1):
            cur_calibration=round(i*args.calibration_interval,2)
            info=torch.load('{}/eval_info_{}.pt'.format(args.save,cur_calibration))
            cur_auroc=round(info['{}_val_auroc_cali_Pred_Tsc'.format(args.in_dataset)].item()*100,4)
            if cur_auroc>max_auroc:
                max_auroc=cur_auroc
                max_calibration=cur_calibration
    calibration_dict['eval']=max_calibration

    max_auroc,max_calibration=0.0,0.0
    if 'ood' in args.inferance:
        for i in range(count+1):
            cur_calibration=round(i*args.calibration_interval,2)
            info=torch.load('{}/ood_info_{}.pt'.format(args.save,cur_calibration))
            print(cur_calibration)
            print(info)
            cur_auroc=round(info['{}_val_auroc_cali_Pred_Tsc'.format(args.in_dataset)].item()*100,4)
            if cur_auroc>max_auroc:
                max_auroc=cur_auroc
                max_calibration=cur_calibration
    calibration_dict['ood']=max_calibration

    torch.save(calibration_dict,'{}/calibration.pt'.format(args.save))


    out_datasets=['out','val','Imagenet','Imagenet_resize','LSUN','LSUN_resize','iSUN']

    if 'eval' in args.inferance:    
        max_calibration=round(torch.load('{}/calibration.pt'.format(args.save))['eval'],2)
        # calibration_list=[max_calibration]
        # calibration_list.append(0.0)
        # calibration_list.append(1.0)
        calibration_list=[round(i*0.05,2) for i in range(21)]
        for item in calibration_list:
            args.calibration=item
            if 'cal_score' in args.excute_list:
                ensemble(args,model_list,out_datasets)
            if 'cal_metric' in args.excute_list:
                cal_metric(args,['tnr95','auroc'],out_datasets)
            if 'print_info' in args.excute_list:
                print_info(args)

    if 'ood' in args.inferance:    
        max_calibration=round(torch.load('{}/calibration.pt'.format(args.save))['ood'],2)
        # calibration_list=[max_calibration]
        # calibration_list.append(0.0)
        # calibration_list.append(1.0)
        calibration_list=[round(i*0.05,2) for i in range(21)]
        for item in calibration_list:
            args.calibration=item
            if 'cal_score' in args.excute_list:
                ensemble(args,model_list,out_datasets)
            if 'cal_metric' in args.excute_list:
                cal_metric(args,['tnr95','auroc'],out_datasets)
            if 'print_info' in args.excute_list:
                print_info(args)
    if 'plot_fig' in args.excute_list:
        plt_fig(args)
    if 'wr' in args.excute_list:
        write_to_excel(args)
    if 'cal_acc' in args.excute_list:
        get_acc(args,model_list)