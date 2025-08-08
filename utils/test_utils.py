import time
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import logging

def load_map(root):
    with open(root, 'r') as f:
        label_map = eval(f.read())
    return label_map


def validate(val_loader, model, criterion, print_freq=10, device='cuda:0', class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    imagenet100_map = load_map('/afs/andrew.cmu.edu/usr14/furix/furix/Public/Labels.txt')
    imagenet1000_map = load_map('/afs/andrew.cmu.edu/usr14/furix/furix/Public/imagenet1000_clsidx_to_labels.txt')
    reverse_imagenet100_map = {v: k for k, v in imagenet100_map.items()}
    reverse_imagenet1000_map = {v: k for k, v in imagenet1000_map.items()}
        

    n_classes_1000 = 1000
    translation = torch.full((n_classes_1000,), -1, device=device, dtype=torch.long)
    
    # Map ImageNet-1000 indices to ImageNet-100 indices
    for k, v in reverse_imagenet1000_map.items():
        if k in reverse_imagenet100_map:
            # k is class name, v is 1000-class index
            translation[v] = class_to_idx[reverse_imagenet100_map[k]]
    

    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            # Forward pass - get 1000-class predictions
            output_1000 = model(data)  # shape: [batch_size, 1000]
            
            # Translate to 100 classes efficiently
            batch_size = output_1000.shape[0]
            output_100 = torch.zeros((batch_size, 100), device=device)
            
            # Use mask to copy only valid classes
            valid_indices = translation != -1
            output_100[:, translation[valid_indices]] = output_1000[:, valid_indices]
            
            output = output_100
        
        # Translate the target to 1000 class
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             i,
                             len(val_loader),
                             batch_time=batch_time,
                             loss=losses,
                             top1=top1,
                             top5=top5,
                         ))
    val_end_time = time.time()
    logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.3f} Time {time:.3f}'.
                 format(top1=top1, top5=top5, time=val_end_time - val_start_time, losses=losses))

    return losses.avg, top1.avg, top5.avg


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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
