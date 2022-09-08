import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from configs import *
from get_data import *
from utils import *


# import setproctitle

def train(args,gt):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists('{}/models'.format(args.output_path)):
        os.mkdir('{}/models'.format(args.output_path))
    save_file_name='{}/models/seed{}_epochs{}_s{:.6f}_proportion{}'.format(args.output_path,args.seed,args.epochs,args.smoothness,args.proportion)
    if os.path.exists('{}.pth'.format(save_file_name)):
        return
    # args.train_batch_size=64 if args.net.find('dense')>-1 else 128
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

    if os.path.exists('{}_train.csv'.format(save_file_name)):
        os.remove('{}_train.csv'.format(save_file_name))
    if os.path.exists('{}_test.csv'.format(save_file_name)):
        os.remove('{}_test.csv'.format(save_file_name))
    trainF = open('{}_train.csv'.format(save_file_name), 'w')
    testF = open('{}_test.csv'.format(save_file_name), 'w')
    for epoch in range(1, args.epochs + 1):
        adjust_opt(args.opt, optimizer, epoch,args)
        iter_train(args, epoch, net, trainLoader, optimizer, trainF,gt)
        iter_test( epoch, net, testLoader, testF)
        if epoch==args.epochs:
            # torch.save(net, os.path.join(args.save, 'latest-{}.pth'.format(epoch)))
            if torch.cuda.device_count()>1:
                torch.save({'state_dict': net.module.state_dict()}, '{}.pth'.format(save_file_name))
            else:
                torch.save({'state_dict': net.state_dict()}, '{}.pth'.format(save_file_name))
            
    trainF.close()
    testF.close()

def iter_train(args, epoch, net, trainLoader, optimizer, trainF,gt):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()
        output = net(data)[0]
        loss_ce,loss_wt=torch.tensor(0),torch.tensor(0)
        if args.smoothness==0.0:
            loss_ce=F.nll_loss(output, target)
            loss=loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # y=F.one_hot(target,num_classes=args.class_num).float()
            # for i in range(len(y)):
            #     if y[i][0]==1:
            #         y[i]=gt[0].cuda()
            #     elif y[i][1]==1:
            #         y[i]=gt[1].cuda()
            # loss_ce = -(y * output).sum(dim=1).mean()
            wt=torch.tensor(args.smoothness,dtype=float).cuda()
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
                                                                                                    

        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        # print('{} \tTrain Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss_ce: {:.4f}\tLoss_wt: {:.10f}\tError: {:.2f}\t lr:{:.3f}'.format(
        #     '{}/PreTrain/seed{}_s{:.6f}'.format(args.output_path,args.seed,args.smoothness),partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
        #     loss_ce.item(),loss_wt.item(), err,optimizer.param_groups[0]['lr']))

        trainF.write('{},{},{},{}\n'.format(partialEpoch,loss_ce.item(),loss_wt.item(), err))
        trainF.flush()
    print('{},{} \tTrain Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss_ce: {:.4f}\tLoss_wt: {:.10f}\tError: {:.2f}\t lr:{:.3f}'.format(
        time.asctime(time.localtime(time.time())),'{}/models/seed{}_s{:.6f}'.format(args.output_path,args.seed,args.smoothness),
        partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),loss_ce.item(),loss_wt.item(),err,optimizer.param_groups[0]['lr']))

def iter_test(epoch, net, testLoader, testF):
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
    print('{},Test set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)'.format(
        time.asctime(time.localtime(time.time())),test_loss, incorrect, nTotal, err))

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




