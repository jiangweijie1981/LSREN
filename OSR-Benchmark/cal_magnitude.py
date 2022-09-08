
import torch
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from configs import *
from get_data import *
from utils import *



def main(args):
    cal_magnitude(args)
    file_path="{}/magnitude.pt".format(args.save)
    magenitude=torch.load(file_path)
    with open("{}/magnitude.txt".format(args.save), 'wt') as f:
        print(magenitude,file=f)



def cal_magnitude(args):
    info=dict()
    info['magnitude']=0
    torch.save(info,"{}/info.pt".format(args.save))
    return
    t0 = time.time()
    dataloader=get_dataloader(args,'train')
    # test modify 5-1
    begin_magnitude,end_magnitude,interval_magnitude=0.0,0.001,0.0001
    file_path=args.save
    model_path="{}/latest-{}.pth".format(file_path,args.epochs)
    net=load_net(args,model_path)
    
    criterion = nn.CrossEntropyLoss()
    test_times=(int)((end_magnitude-begin_magnitude)/interval_magnitude+1)
    max_magnitude, max_value=begin_magnitude,-10000
    for i in range (test_times):
        args.magnitude=np.round(begin_magnitude+interval_magnitude*i,4)
        print("\nModel:{}, Processing {} images, magnitude:{:.4f}".format(file_path, args.in_dataset,args.magnitude))
        Msp,Msp_Odin,total_length=0,0,0
        for j, data in enumerate(dataloader):
            # test modify 5-2
            if j>=10: break
            images, _ = data
            inputs = Variable(images.cuda(), requires_grad = True)
            outputs = net(inputs)[2]
            softmax_output=F.softmax(outputs/1000,dim=1)
            Msp_batch = torch.max(softmax_output,dim=1)[0]
            Msp+=torch.sum(Msp_batch).item()
            maxIndexTemp = torch.argmax(softmax_output,dim=1)
            loss = criterion(softmax_output, maxIndexTemp)
            loss.backward()
            # Normalizing the gradient to binary in {0, 1}
            gradient =  torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
            gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
            gradient[:,2] = (gradient[:,2])/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data,  -args.magnitude, gradient)
            with torch.no_grad():
                outputs = net(Variable(tempInputs))[2]
            # Calculating the confidence after adding perturbations
            softmax_output=F.softmax(outputs/1000,dim=1)
            Msp_Odin_batch = torch.max(softmax_output,dim=1)[0]
            Msp_Odin += torch.sum(Msp_Odin_batch).item()
            total_length+=len(images)
        cur_value=(Msp_Odin-Msp)/total_length
        if cur_value>max_value:
            max_magnitude=args.magnitude
            max_value=cur_value
        print('max_magnitude:{:.4f}, max_value:{:.6f}, cur_value:{:.6f}, {:.1f} seconds used.'.format(max_magnitude,max_value,cur_value,time.time()-t0))

    info=dict()
    info['magnitude']=max_magnitude
    torch.save(info,"{}/info.pt".format(file_path))




if __name__ == '__main__':
    args=Config().data
    set_seed(args.seed)
    smooth_list=get_smooth_list(args)
    if args.train_type=='LSR':
        for i in range(len(smooth_list)):
            args.smooth=smooth_list[i]
            args.save = '../models/{}/{}_{}{}/{}_{}/seed{}_epochs{}_s{:.2f}'.format(
                args.net,args.in_dataset,args.noise_type,args.noise_ratio,
                args.train_type,args.ensemble_num,args.seed,args.epochs,args.smooth)
            main(args)
    if args.train_type=='ILSR':
        for i in range(len(args.penalty_list)):
            args.penalty=args.penalty_list[i]
            for j in range(len(smooth_list)):
                args.smooth=smooth_list[j]
                args.save = '../models/{}/{}_{}{}/{}_{}_{}/seed{}_epochs{}_s{:.2f}'.format(
                    args.net,args.in_dataset,args.noise_type,args.noise_ratio,
                    args.train_type,args.ensemble_num,args.penalty,args.seed,args.epochs,args.smooth)
                main(args)