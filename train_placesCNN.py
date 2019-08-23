# this code is modified from the pytorch example code: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# after the model is trained, you might use convert_model.py to remove the data parallel module to make the model as standalone weight.
#
# Bolei Zhou

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import wideresnet
import pdb
import SENet
import progressbar
from PIL import Image
import pandas as pdb
import csv
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')
parser.add_argument('--dataset',default='places365',help='which dataset to train')
device_ids = [2,3]
ini_device = 2
best_prec1 = 0



def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.lower().startswith('wideresnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model  = wideresnet.resnet50(num_classes=args.num_classes)
    elif args.arch.lower().startswith('se'):
        model  = SENet.se_resnet50(num_classes=args.num_classes)
    elif args.arch.lower().startswith('new'):
        model  = SENet.se_resnet(num_classes=args.num_classes)
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes)

    print('#parameters:', sum(param.numel() for param in model.parameters()))

    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, device_ids)
        model.to(ini_device)
    else:
        model = torch.nn.DataParallel(model, device_ids).to(ini_device)



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        print(model)
    
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().to(ini_device)

    if args.evaluate:
        #validate(val_loader, model, criterion)
        
        trainMidOutputs = getMidOutputs(train_loader, model, 15)
        valMidOutputs = getMidOutputs(val_loader, model, 15)
        fcModel = SENet.simpleFcNet(365)
        fcModel = torch.nn.DataParallel(fcModel, device_ids).cuda()
        optimizer = torch.optim.SGD(fcModel.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        num_epochs = 180
        print('init train acc')
        validate(trainMidOutputs, fcModel, criterion)
        print('init val acc')
        validate(valMidOutputs, fcModel, criterion)

        trainFc(trainMidOutputs, 1e-5, num_epochs, criterion, optimizer, fcModel, valMidOutputs)

        validate(trainMidOutputs, fcModel, criterion)
        validate(valMidOutputs, fcModel, criterion)
        
        return
    else:

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, args.arch.lower())



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    bar = progressbar.progressbar(len(train_loader))
    end = time.time()
    start = time.perf_counter()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
 
        target = target.to(ini_device ,non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #bar.clear()
        if i % args.print_freq == 0:
            print('\rEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        #bar.output(i+1)
    #print()
    print('Epoch waste time {}s'.format(time.perf_counter()- start) )

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(ini_device, non_blocking=True)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            '''
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


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
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
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


def getErrorImgInfo(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    pred = pred.t()
    infos = []
    for k in topk:
        mask = (correct[:k].sum(0) == 0)
        errorImgIndex = torch.arange(batch_size)[mask]
        infos.append((errorImgIndex,pred[mask,:k],target[mask]))
    return infos

def getClassNameByTensor(checkTensor, dataSet):
    msg = ''
    for i in range(checkTensor.shape[0] ):
        msg += dataSet.classes[checkTensor[i] ]  + ' '
    return msg




def saveTensorAsImg(path, tensor):
    unloader = transforms.ToPILImage()
    tensorImage = unloader(tensor.cpu().clone())
    tensorImage.save(path)

def checkErrorImage(val_loader, model, criterion):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
 
    end = time.time()
    global args
    valdir = os.path.join(args.data, 'val')
    dataSetRootLen = len(valdir)
    valDataSet = datasets.ImageFolder(valdir)

    classNum = len(valDataSet.classes)
    confuseMat1 = torch.zeros(classNum, classNum)
    confuseMat5 = torch.zeros(classNum, classNum)
   

    errorMsgDir = '../errorMsg/'
    errorImgFile1 = open(errorMsgDir+'errorImgFile1.txt','w')
    errorImgFile5 = open(errorMsgDir+'errorImgFile5.txt','w') 
    statisticFile = open(errorMsgDir+'statistics.csv', 'w', newline='')
    statisticFile1 = open(errorMsgDir+'statistics1.csv', 'w', newline='')
    # 设定写入模式
    statistic_write = csv.writer(statisticFile, dialect='excel')
    statistic_write1 = csv.writer(statisticFile1, dialect='excel')
    # 写入具体内容
    csv_header=['className', 'top1ErrorNum', 'top1top3ErrorClassName', 'top1top3value',
                'top5ErrorNum', 'top5top3ErrorClassName', 'top5top3value']
    statistic_write.writerow(csv_header)
    statistic_write1.writerow(csv_header)
    bar = progressbar.progressbar(len(val_loader))
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            errorInfos1, errorInfos5 = getErrorImgInfo(output.data, target, topk=(1, 5))
            #top5Predict是top5预测结果，labelIndex是标签下标
            
            #top5处理

            for j in range(errorInfos5[0].size(0)):
                top5Predict = errorInfos5[1][j]
                labelIndex = errorInfos5[2][j].view(-1)
                confuseMat5[labelIndex.item(),top5Predict] += 1 
               
                imgIndex = errorInfos5[0][j] + i * 256
                top5Result = getClassNameByTensor(top5Predict, valDataSet)
                realResult = getClassNameByTensor(labelIndex, valDataSet)
                
                errorImgName = valDataSet.samples[imgIndex][0][dataSetRootLen:]
                errorImgFile5.write(errorImgName + ' top5: ' + top5Result + 'real: ' + realResult + '\n')
            
            for j in range(errorInfos1[0].size(0)):
                top1Predict = errorInfos1[1][j]
                labelIndex = errorInfos1[2][j].view(-1)
                confuseMat1[labelIndex.item(),top1Predict] += 1 
                
                imgIndex = errorInfos1[0][j] + i * 256
                top1Result = getClassNameByTensor(top1Predict, valDataSet)
                realResult = getClassNameByTensor(labelIndex, valDataSet)
                
                errorImgName = valDataSet.samples[imgIndex][0][dataSetRootLen:]
                errorImgFile1.write(errorImgName + ' top1: ' + top1Result + 'real: ' + realResult + '\n')

            #print(errorMask1.shape,errorMask1)

            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            end = time.time()
            bar.output(i+1)
    print()
    saveTensorAsImg(errorMsgDir+'confuseMat5Image.jpg', confuseMat5)
    saveTensorAsImg(errorMsgDir+'confuseMat1Image.jpg', confuseMat1)

    for i in range(classNum):
        nowClassName = valDataSet.classes[i]
        top1top3value, pred = confuseMat1[i].topk(3)
        top1ErrorNum = confuseMat1[i].sum().item()
        top1top3ErrorClassName = getClassNameByTensor(pred, valDataSet)

        top5top3value, pred = confuseMat5[i].topk(3)
        top5ErrorNum = confuseMat5[i].sum().item() / 5
        top5top3ErrorClassName = getClassNameByTensor(pred, valDataSet)
        row = [nowClassName, top1ErrorNum, top1top3ErrorClassName, str(top1top3value),
                top5ErrorNum, top5top3ErrorClassName, str(top5top3value)]
        statistic_write.writerow(row)
    for i in range(classNum):
        nowClassName = valDataSet.classes[i]
        top1top3value, pred = confuseMat1[:,i].topk(3)
        top1ErrorNum = confuseMat1[:,i].sum().item()
        top1top3ErrorClassName = getClassNameByTensor(pred, valDataSet)

        top5top3value, pred = confuseMat5[:,i].topk(3)
        top5ErrorNum = confuseMat5[:,i].sum().item() 
        top5top3ErrorClassName = getClassNameByTensor(pred, valDataSet)
        row = [nowClassName, top1ErrorNum, top1top3ErrorClassName, str(top1top3value),
                top5ErrorNum, top5top3ErrorClassName, str(top5top3value)]
        statistic_write1.writerow(row)

    errorImgFile5.close()
    errorImgFile1.close()
    statisticFile.close()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def getMidOutputs(loader, model, topkNum=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    midOutputs = [] 
    bar = progressbar.progressbar(len(loader))
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input)
            output = output.cpu()
            output = F.softmax(output, dim=1)
            if(topkNum is not None):
                _, pred = output.topk(topkNum, 1)
                mask = torch.zeros(output.shape)
                for j in range(pred.shape[0]):
                    mask[j, pred[j]] = 1
                output *= torch.FloatTensor(mask)

            midOutputs.append( (output.cpu(), target.cpu()) )
            # measure accuracy and record loss
            batch_time.update(time.time() - end)
            end = time.time()
            bar.output(i+1)

        print()
    return midOutputs

def trainFc(midOutputs, learningRate, num_epochs, criterion, optimizer, fcModel, validateMidoutputs=None):
    curr_lr = learningRate
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total = 0
        correct = 0

        for i, (images, target) in enumerate(midOutputs):
            fcModel.train()
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(target)

            outputs = fcModel(input_var)
            loss = criterion(outputs, target_var)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target_var).sum().item()
            total += target_var.size(0)
        if(validateMidoutputs is not None):
            validate(validateMidoutputs, fcModel, criterion)
        random.shuffle(midOutputs)
        epoch_loss /= len(midOutputs)


        train_acc = correct / total * 100
        
        print ("epoch [{}/{}], avg_loss: {:.4f}, train_acc: {:.4f}"
               .format(epoch+1, num_epochs, epoch_loss, train_acc))

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': fcModel.state_dict(),
                'best_prec1': best_prec1,
            }, True, filename='nnModel')
        if (epoch+1) % 60 == 0:

            curr_lr /= 10
            update_lr(optimizer, curr_lr)


def validateFc(valMidOutputs, fcModel, criterion):
    epoch_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(valMidOutputs):

            fcModel.eval()

            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(target)

            outputs = fcModel(input_var)
            loss = criterion(outputs, target_var)
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target_var).sum().item()
            total += target_var.size(0)
        epoch_loss /= len(valMidOutputs)
        val_acc = correct / total * 100
        print()
        print ("validate avg_loss: {:.4f}, val_acc: {:.4f}"
               .format(epoch_loss, val_acc))


if __name__ == '__main__':
    main()

