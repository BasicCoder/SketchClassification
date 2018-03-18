from __future__ import print_function, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
from SketchANetModel import SketchANetModel
from AlexNetModel import AlexNetModel
from ResNetModel import ResNetModel

parser = argparse.ArgumentParser(description='PyTorch Sketch Me That Shoe Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='Adm weight decay')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--print-freq', '-p', default=15, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--classes', type=int, default=419,
                    help='number of classes')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNetModel', type=str,
                    help='name of experiment')
parser.add_argument('--normalize_feature', default=False, type=bool,
                    help='normalize_feature')

best_acc = 0


def to_scalar(vt):
    """Transform a length-1 pytorch Variable or Tensor to scalar.
    Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]),
    then npx = tx.cpu().numpy() has shape (1,), not 1."""
    if isinstance(vt, Variable):
        return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        return vt.cpu().numpy().flatten()[0]
    raise TypeError('Input should be a variable or tensor')


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    ###### DataSet ######
    sketch_dir =  r"/home/bc/Work/Database/TU-Berlin sketch dataset/png"
    # sketch_dir = r"/home/bc/Work/Database/Dogs_Cats/catdog/train"
    train_dataset = datasets.ImageFolder(
        sketch_dir,
        transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_dir = r"/home/bc/Work/Database/Dogs_Cats/catdog/val"
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                # std=[0.229, 0.224, 0.225])
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )
    ###### Model ######

    # snet = SketchANetModel(num_classes=250)
    # snet = AlexNetModel(num_classes=250)
    snet = ResNetModel(num_classes=250)
    print(snet)
    if args.cuda:
        snet.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec']
            snet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ###### Criteria ######
    id_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(snet.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    n_parameters = sum([p.data.nelement() for p in snet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, snet, id_criterion, optimizer, epoch)
        # evaluate on validation set
        # prec1 = test(test_loader, snet, id_criterion, epoch)

        # remember best Accuracy and save checkpoint
        #is_best = prec1 > best_acc
        is_best = True
        #best_acc = max(prec1, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': snet.state_dict(),
            'best_prec': best_acc,
        }, is_best)

def train(train_loader, snet, id_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    snet.train()

    end = time.time()
    for batch_indx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        # compute output
        _, output = snet(input)

        # print(output.data[0])
        loss = id_criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm(snet.parameters(), 100.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_indx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_indx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

def test(test_loader, snet, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    snet.eval()

    end = time.time()
    for batch_indx, (input, target) in enumerate(test_loader):
        if args.cuda:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        # compute output
        _, output = snet(input)
        output = snet(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_indx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_indx, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
