import time
import torch
import torch.nn.functional as F
import math

# def adjust_learning_rate(optimizer,  init_lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = init_lr * 0.1 
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
def adjust_learning_rate(optimizer, epoch, step, len_iter , args):

    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5 ** factor)
    
    elif args.lr_type == 'cos':  # cos without warm-up
        ###APIB
        # lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.fine_epochs))
        # lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch) / (args.fine_epochs)))
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.fine_epochs - 5)))


    elif args.lr_type == 'exp':
        
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    #Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if step == 0:
        print('learning_rate : ', lr)
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion , args, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda()
            # input_var = input.cuda()
            # target_var = target.cuda()
            
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)
            
            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target)[0]
            prec1 , prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item() , input.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1 , top5 = top5))

    print(' * Prec@1 {top1.avg:.3f} , Prec@5 {top5.avg:.3f}'
          .format(top1=top1 , top5 = top5))

    return top1.avg , top5.avg