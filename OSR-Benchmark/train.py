import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
from configs import *
from get_data import *
from utils import *
from cal_magnitude import *
from cal_score import *
from cal_metric import *

# import setproctitle

def train(args):
    if os.path.exists('{}/latest-{}.pth'.format(args.save,args.epochs)):
        return
    if os.path.exists(args.save):
        shutil.rmtree(args.save)

    os.makedirs(args.save, exist_ok=True)

    args.train_batch_size=64 if args.net.find('dense')>-1 else 128
    trainLoader,testLoader=get_dataloader(args,'train'),get_dataloader(args,'test')

    net=get_net(args)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,momentum=0.9, weight_decay=args.decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)
    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    for epoch in range(1, args.epochs + 1):
        adjust_opt(args.opt, optimizer, epoch,args)
        iter_train(args, epoch, net, trainLoader, optimizer, trainF)
        iter_test(args, epoch, net, testLoader, optimizer, testF)
        if epoch==args.epochs:
            # torch.save(net, os.path.join(args.save, 'latest-{}.pth'.format(epoch)))
            if torch.cuda.device_count()>1:
                torch.save({'state_dict': net.module.state_dict()}, '{}/latest-{}.pth'.format(args.save,epoch))
            else:
                torch.save({'state_dict': net.state_dict()}, '{}/latest-{}.pth'.format(args.save,epoch))
            
    trainF.close()
    testF.close()

def iter_train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()
        output,output_smth,output_logit = net(data)
        loss_ce,loss_wt=torch.tensor(0),torch.tensor(0)
        if args.train_type=='LSR' and args.smooth==0:
            loss_ce=F.nll_loss(output, target)
            loss=loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif args.train_type=='ILSR':
            wt=torch.sigmoid(output_smth)
            wt_expand=wt.expand(output.size(0),args.class_num)
            smoothing_label=wt_expand/(args.class_num-1)
            y=F.one_hot(target,num_classes=args.class_num).float()
            wt_y=y-torch.mul(wt_expand,y)
            smoothing_label=torch.relu(smoothing_label-y)
            smoothing_label=smoothing_label+wt_y
            # loss_ce = -(smoothing_label.detach() * output).sum(dim=1).mean()
            loss_ce = -(smoothing_label * output).sum(dim=1).mean()
            loss_wt=wt.mean()
            # loss_wt=torch.abs(wt.mean()-args.smooth)
            loss=loss_ce+args.penalty*loss_wt
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            wt=torch.tensor(args.smooth,dtype=float).cuda()
            wt_expand=wt.expand(output.size(0),args.class_num)
            smoothing_label=wt_expand/(args.class_num-1)
            y=F.one_hot(target,num_classes=args.class_num).float()
            wt_y=y-torch.mul(wt_expand,y)
            smoothing_label=torch.relu(smoothing_label-y)
            smoothing_label=smoothing_label+wt_y
            loss_ce = -(smoothing_label * output).sum(dim=1).mean()
            loss=loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                                                                                                    
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)

        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('{} \tTrain Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss_ce: {:.4f}\tLoss_wt: {:.10f}\tError: {:.2f}\t lr:{:.3f}'.format(
            args.save,partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss_ce.item(),loss_wt.item(), err,optimizer.param_groups[0]['lr']))

        trainF.write('{},{},{},{}\n'.format(partialEpoch,loss_ce.item(),loss_wt.item(), err))
        trainF.flush()

def iter_test(args, epoch, net, testLoader, optimizer, testF):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        incorrect = 0
        for data, target in testLoader:
            data, target = data.cuda(), target.cuda()
            output = net(data)[0]
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch,args):
    if optAlg == 'sgd':
        if epoch < args.epochs//2: lr = 1e-1
        elif epoch == args.epochs//2: lr = 1e-2
        elif epoch == args.epochs-args.epochs//4: lr = 1e-3
        else: return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



if __name__=='__main__':
    args=Config().data
    set_seed(args.seed)
    smooth_list=get_smooth_list(args)
    if args.train_type=='LSR':
        for i in range(len(smooth_list)):
            args.smooth=smooth_list[i]
            args.save = '../models/{}/{}_{}/{}_{}/seed{}_epochs{}_s{:.2f}'.format(
                args.net,args.in_dataset,args.set_id,
                args.train_type,args.ensemble_num,args.seed,args.epochs,args.smooth)
            train(args)
            cal_magnitude(args)
            cal_score(args)
            cal_metric(args,['tnr95','auroc'])
            print_info(args)
    if args.train_type=='ILSR':
        for i in range(len(args.penalty_list)):
            args.penalty=args.penalty_list[i]
            for j in range(len(smooth_list)):
                args.smooth=smooth_list[j]
                args.save = '../models/{}/{}_{}/{}_{}_{}/seed{}_epochs{}_s{:.2f}'.format(
                    args.net,args.in_dataset,args.set_id,
                    args.train_type,args.ensemble_num,args.penalty,args.seed,args.epochs,args.smooth)
                train(args)
                cal_magnitude(args)
                cal_score(args)
                cal_metric(args,['tnr95','auroc'])
                print_info(args)