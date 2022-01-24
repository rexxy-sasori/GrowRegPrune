import os
import time
from copy import deepcopy

import torch
from nnutils.training_pipeline import accuracy_evaluator
from torch import nn


def finetune(pruner, lr_ft, num_epochs, update_freq, logger, output_dir):
    logger.info(f"Registering mask to reged model")
    register_mask(pruner)

    sparse_model = deepcopy(pruner.model)
    logger.info(f"Finishing loading reged model")

    trainloader = pruner.trainloader
    testloader = pruner.testloader
    criterion = pruner.criterion
    optimizer = pruner.optimizer
    lr_scheduler = PresetLRScheduler(lr_ft)
    logger = pruner.logger
    device = pruner.device

    tune_model(criterion, device, logger, lr_scheduler, num_epochs, optimizer, output_dir, sparse_model, testloader,
               trainloader, update_freq)


def tune_model(criterion, device, logger, lr_scheduler, num_epochs, optimizer, output_dir, sparse_model, testloader,
               trainloader, update_freq):
    best_acc = 0
    for epoch in range(num_epochs):
        lr = lr_scheduler(optimizer, epoch)
        logger.info("==> Set lr = %s @ Epoch %d " % (lr, epoch))
        train(trainloader, sparse_model, criterion, optimizer, epoch, logger, device, update_freq)
        acc, loss = accuracy_evaluator.eval(sparse_model, device, testloader, criterion, print_acc=False)
        logger.info(f"TEST ACC {acc} at Iter {epoch}")

        if acc > best_acc:
            torch.save(
                {
                    'state_dict': sparse_model.state_dict(),
                    'test_acc': acc,
                    'test_loss': loss,
                    'iteration': epoch,
                },
                os.path.join(output_dir, 'finetune_ckpt.pt')
            )
            best_acc = acc


def register_mask(pruner):
    for name, target in pruner.target_layers.items():
        setattr(target.layer, 'mask', target.kp_mask)


def train(train_loader, model, criterion, optimizer, epoch, logger, device, update_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        logger,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        apply_mask_forward(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % update_freq == 0:
            progress.display(i)


def apply_mask_forward(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data.mul_(module.mask)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, logger, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class PresetLRScheduler(object):
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = {}
        for k, v in decay_schedule.items():  # a dict, example: {"0":0.001, "30":0.00001, "45":0.000001}
            self.decay_schedule[int(k)] = v
        # print('Using a preset learning rate schedule:')
        # print(self.decay_schedule)

    def __call__(self, optimizer, e):
        epochs = list(self.decay_schedule.keys())
        epochs = sorted(epochs)  # example: [0, 30, 45]
        lr = self.decay_schedule[epochs[-1]]
        for i in range(len(epochs) - 1):
            if epochs[i] <= e < epochs[i + 1]:
                lr = self.decay_schedule[epochs[i]]
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
