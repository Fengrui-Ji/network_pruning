import argparse
import os
from copy import deepcopy
import shutil
import time
from numpy import zeros_like
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from my_admm import *
import torchvision.models as models
from optimizer import *
from pathlib import Path
from datasets import *
import gc
from compute_flops import *
from collections import OrderedDict
from select_channel import *
from utils import *
from models.built_model import *
from channel_pruning_resnet import *



def cos_lr(optimizer , lr , epoch , total_epoch):
    lr = 0.5 * lr * (1 + math.cos(math.pi * (epoch - 5) / (total_epoch - 5)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

        
def change_learning_rate(optimizer,  init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr


def set_args():
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--fine-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs for finetune')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--re-lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--device', default='0', type=str, metavar='M',
                    help='GPU id')
    parser.add_argument('--model',  required=True, choices=["resnet18", "resnet20","resnet50", "vgg19", "vgg16" , "resnet56","resnet101"] ,
                    help='model')
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tiny_imagenet", "imagenet"])
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
    parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
    parser.add_argument('--lamb', type=float, default=1e-4,
                    help='file name to save')
    parser.add_argument('--seed', type=int, default=random.randint(1, 1e3),
                    help='file name to save')
    parser.add_argument('--data', type=float, default=1.,
                    help='how many data to compress the model')
    parser.add_argument('--ratio', type=float, default=0.9,
                    help='file name to save')
    parser.add_argument('--p', type=int, default=5,
                    help='p < 1 norm')
    parser.add_argument('--M', type=int, default=-1,
                    help='M value')
    parser.add_argument('--rho', type=float, default=1.5e-3,
                    help='file name to save')
    parser.add_argument('--adjust', type=float, default=0.3,
                    help='adjust layer')
    parser.add_argument('--compressed-epoch', type=int, default=3,
                    help='file name to save')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R',
                        help='number of epochs to retrain (default: 3)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E',
                        help='adam epsilon (default: 1e-8)')
    parser.add_argument('--sp','--structured-pruning' , action='store_true', default=False,
                        help='structed pruning?')
    parser.add_argument('--up','--unstructured-pruning', action='store_true', default=False,
                        help='unstructed pruning?')
    parser.add_argument('--lr_type', type=str,  default='cos', help='lr type')
    args = parser.parse_args()
    return args

def fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    
    args = set_args()
    print(args)
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    GPU_id = 'cuda:' + args.device
    device = torch.device(GPU_id)
    
    print("built model")
    model = model_factory(args.model, args.dataset , False)
    model = model.to(device)
    
    print("finish built model")
    print(model)
    

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    seed = args.seed
    print('seed : ' , seed)
    fix(seed)
    
    
    data_path = Path('/root/data')
    if args.dataset == 'cifar10':
        num_classes, train_dataset, test_dataset = get_cifar10(data_path)
    elif args.dataset == 'cifar100':
        num_classes, train_dataset, test_dataset = get_cifar100(data_path)
    elif args.dataset == 'imagenet':
        num_classes, train_dataset, test_dataset = get_imagenet(data_path)
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=32, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
       test_dataset, batch_size=128, shuffle=False,num_workers=4, pin_memory=False)

   
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_prec1 = 0
    
    # load pretrained model
    model = models.resnet50(pretrained=True)
    temp = torch.load('/root/pretrain_model/resnet50.pth')['model_state_dict']
    model.load_state_dict(temp)
    
    
    #compress model using L_p(p < 1) regularization
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    compress_model(train_loader, model, criterion, optimizer, args.compressed_epoch , device , args , val_loader)
    if args.sp:
        shape = (1,3,224,224)
        sparse_model = get_channel_mag(model , shape , args , device)
        torch.compile(sparse_model)
        optimizer = torch.optim.SGD(sparse_model.parameters(), lr = 1e-1,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
        sparse_model.to(device)
        best_prec1 = 0
        if args.fine_epochs == 30:
            lr_scheduler = [10,20]
        elif args.fine_epochs == 90:
            lr_scheduler = [30,60,80]
        else:
            lr_scheduler = [140 , 240]
              
        optimizer = torch.optim.SGD(sparse_model.parameters(), lr = args.re_lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
        torch.set_float32_matmul_precision('high')
        for epoch in range(args.fine_epochs):
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
            train(train_loader, sparse_model, criterion, optimizer, epoch , args , device)
            prec1 , prec5 = validate(val_loader, sparse_model, criterion , args, device)
            if prec1 > best_prec1:
                best_prec1 = prec1
            print('best : ' , best_prec1)
            name = '/root/finetune_model/' + str(args.model) + '_' + str(epoch) +'.pth'
            torch.save({'model_state_dict': sparse_model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()},
                  name)
        print('finish!')
        
    if args.up:
        mask , param_name_list = our_apply_prune(model, device, args.ratio)
        print_prune(model)
        Prune_optimizer = torch.optim.SGD(model.parameters(), lr = 0.01,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
        re_train(train_loader, mask , param_name_list , model, criterion, Prune_optimizer , 100 , device , scheduler , val_loader , args)

    

def compress_model(train_loader , model , criterion , optimizer , epochs , device , args , val_loader):
    """
        Run one train epoch
    """ 
    end = time.time()
    param_name = get_param(model)
    X , Z , U = initialize_X_Z_and_U(model , param_name)
    p = args.p
    for epoch in range(epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            optimizer.zero_grad()
            target_var = target.to(device)
            input_var = input.to(device)
            output = model(input_var)
            loss = criterion(output, target_var)
            local = 0
            for name , param in model.named_parameters():
                if name.split('.')[-1] == "weight":
                    loss += 0.5 *  args.rho * (param - Z[local : local + param.numel()].reshape(param.shape) + U[local : local + param.numel()].reshape(param.shape)).norm()
                    local = local + param.numel()
                    
            

            loss.backward()
            optimizer.step()

              
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % 50 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


        X = get_X(model , param_name)
        Xk = X
        Old_Z = Z.detach()
        if p == 1:
            Z = update_Z_l1(args , X ,U)
        elif p > 1:
            Z = update_Z_lp(args , X, Z ,U , p)
        else:
            Z = update_Z_l0(args , X ,U)
        print('Z : ' , Z)
        U = update_U(U , X , Z)
        print('pir : ' , (X - Z).norm())
        prec1 = validate(val_loader, model, criterion , args , device)
        name = '/root/compress_model/' + str(args.model) + str(args.lamb) + '_' + str(args.lr) + str(epoch) +'.pth'
        torch.save({'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()},
                  name)



def re_train(train_loader, mask , param_name_list, model, criterion, optimizer, epoch , device , scheduler , val_loader , args):
    """
        Run one train epoch
    """
    

    # switch to train mode
    model.train()

    end = time.time()
    max_prec1 = 0
    for epochs in range(epoch): 
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()       
        if epochs in scheduler:
            adjust_learning_rate(optimizer , optimizer.param_groups[0]['lr'])
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        for i, (input, target_var) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input_var = input.to(device)
            target_var = target_var.to(device)
            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            # criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
            loss = criterion(output, target_var)  
            optimizer.zero_grad()
            loss.backward()
            local = 0
            for name , param in model.named_parameters():
                param.grad.mul_(mask[local])
                local += 1
            optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input_var.size(0))
            top1.update(prec1.item(), input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Re-Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epochs, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

        prec1 = validate(val_loader, model, criterion , args ,device)
        if prec1 >= max_prec1:
            max_prec1 = prec1
        print('max : ' , max_prec1)
        


def train(train_loader, model, criterion, optimizer, epoch , args ,device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    num_iter = len(train_loader)
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        adjust_learning_rate(optimizer, epoch, i, num_iter , args)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        
        
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)




if __name__ == '__main__':
    main()
    
