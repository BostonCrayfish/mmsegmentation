#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import logging
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from mmcv.utils import Config

from moco.moco import loader as moco_loader
from moco.moco import builder_dense_1 as moco_builder

from torch.utils.tensorboard import SummaryWriter

logger_moco = logging.getLogger(__name__)
logger_moco.setLevel(level=logging.INFO)
handler = logging.FileHandler('./log_moco.txt')
handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger_moco.addHandler(handler)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

writer = SummaryWriter('./log')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    default='/export/ccvl11b/cwei/data/ImageNet',
                    help='path to dataset')
parser.add_argument('--device-name', default='ccvl11', help='device name')
parser.add_argument('--head', default='aspp', help='decoder head')
parser.add_argument('--config', default='config_segco_aspp.py', help='config file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num-images', default=1281167, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--scalar-freq', default=100, type=int,
                    help='metrics writing frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # ngpus_per_node = 4
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # for ease running on different devices
    if args.device_name == 'ccvl11':
        data_dir = '/export/ccvl11b/cwei/data/ImageNet'
        config_dir = '/home/feng/work_place/checkpoints/'
    elif args.device_name == 'ccvl8':
        data_dir = '/home/cwei/feng/data/ImageNet'
        config_dir = '/home/cwei/feng/work_place/checkpoints/'
    elif args.device_name == 's2':
        data_dir = '/stor2/wangfeng/ImageNet'
        config_dir = '/stor2/wangfeng/work_place/checkpoints/'
    elif args.device_name == 's5':
        data_dir = '/stor1/user1/data/ImageNet'
        config_dir = '/home/user1/mmsegmentation/configs/my_config'
    elif args.device_name == 's6':
        data_dir = '/sdb1/fidtqh2/data/ImageNet'
        config_dir = '/sdb1/fidtqh2/work_place/scripts/'
    else:
        raise ValueError("missing data directory or unknown device")

    # if args.head == 'fcn':
    #     cfg_file = 'config_segco_fcn.py'
    # elif args.arch.startswith('vit'):
    #     cfg_file = 'config_segco_' + args.arch + '.py'
    # elif args.arch == 'resnet50v1c':
    #     cfg_file = 'config_segco_r50v1c.py'
    # else:
    #     cfg_file = 'config_segco_aspp.py'
    cfg_file = args.config

    cfg = Config.fromfile(config_dir + cfg_file)
    args.gpu = gpu
    args.config_dir = config_dir

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco_builder.MoCo(
        cfg,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # fix parameters in backbone
    # optimizer = torch.optim.SGD(model.module.encoder_q.decode_head.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    if args.arch.startswith('vit'):
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # set different paths to data
    # for ease running on different devices
    traindir = os.path.join(data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_bg = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=1., scale=(0.5, 0.8), ratio=(0.8, 1.25), value=0.)
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco_loader.TwoCropsTransform(transforms.Compose(augmentation)))
    train_dataset_bg = datasets.ImageFolder(
        traindir,
        transforms.Compose(augmentation_bg))

    # load training data in three processes:
    # 1. train_*: foreground image
    # 2. train_*_bg0: background image 0
    # 3. train_*_bg1: background image 1
    # This process is convenient to load data for epochs
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=0)
        train_sampler_bg0 = torch.utils.data.distributed.DistributedSampler(train_dataset_bg, seed=1024)
        train_sampler_bg1 = torch.utils.data.distributed.DistributedSampler(train_dataset_bg, seed=2048)
    else:
        train_sampler = None
        train_sampler_bg0 = None
        train_sampler_bg1 = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    train_loader_bg0 = torch.utils.data.DataLoader(
        train_dataset_bg, batch_size=args.batch_size, shuffle=(train_sampler_bg0 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_bg0, drop_last=True)
    train_loader_bg1 = torch.utils.data.DataLoader(
        train_dataset_bg, batch_size=args.batch_size, shuffle=(train_sampler_bg1 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_bg1, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            train_sampler_bg0.set_epoch(epoch)
            train_sampler_bg1.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train([train_loader, train_loader_bg0, train_loader_bg1], model, criterion, optimizer, epoch, args)
        if epoch % 5 == 4:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader_list, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_m = AverageMeter('Loss_moco', ':.4e')
    loss_s = AverageMeter('Loss_seg', ':.4e')
    acc_moco = AverageMeter('Acc_moco', ':6.2f')
    acc_seg = AverageMeter('Acc_seg', ':6.2f')
    train_loader, train_loader_bg0, train_loader_bg1 = train_loader_list
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_m, loss_s, acc_moco, acc_seg],
        prefix="Epoch: [{}]".format(epoch))

    cre_dense = nn.LogSoftmax(dim=1)
    # cre_dense = nn.Sigmoid()

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((images, _), (bg0, _), (bg1, _)) in enumerate(zip(train_loader, train_loader_bg0, train_loader_bg1)):
        # measure data loading time
        data_time.update(time.time() - end)

        current_bs = images[0].size(0)

        mask_idx_q = torch.where(bg0[:, 0, :, :] == 0.)
        mask_idx_k = torch.where(bg1[:, 0, :, :] == 0.)
        mask_q = torch.zeros((current_bs, 224, 224))
        mask_k = torch.zeros((current_bs, 224, 224))
        mask_q[mask_idx_q[0], mask_idx_q[1], mask_idx_q[2]] = 1.
        mask_k[mask_idx_k[0], mask_idx_k[1], mask_idx_k[2]] = 1.

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            bg0 = bg0.cuda(args.gpu, non_blocking=True)
            bg1 = bg1.cuda(args.gpu, non_blocking=True)
            mask_q = mask_q.cuda(args.gpu, non_blocking=True)
            mask_k = mask_k.cuda(args.gpu, non_blocking=True)

        # generate patched images
        image_q = torch.einsum('bcxy,bxy->bcxy', [images[0], mask_q]) + bg0
        image_k = torch.einsum('bcxy,bxy->bcxy', [images[1], mask_k]) + bg1

        # compute output
        output_moco, output_dense, target_moco, target_dense, mask_dense = model(
            image_q, image_k, mask_q[:, 8::16, 8::16], mask_k[:, 8::16, 8::16])
        loss_moco = criterion(output_moco, target_moco)

        # dense loss of softmax
        # output_dense = output_dense.reshape(output_dense.shape[0], -1)
        # output_dense[torch.where(mask_dense == 0)] = -1e10
        # output_dense_log = (-1.) * cre_dense(output_dense)
        # loss_dense = torch.mean(
        #     torch.mul(output_dense_log, target_dense).sum(dim=1) / target_dense.sum(dim=1))

        # dense loss of softmax, short
        output_dense_log = (-1.) * cre_dense(output_dense)
        output_dense_log = output_dense_log.reshape(output_dense_log.shape[0], -1)
        loss_dense = torch.mean(
            torch.mul(output_dense_log, target_dense).sum(dim=1) / target_dense.sum(dim=1))

        # dense loss of softmax, k_pos_avg
        # output_dense_log = (-1.) * cre_dense(output_dense)
        # loss_dense = torch.mul(output_dense_log, target_dense).sum(dim=1).mean() / target_dense.sum()

        # dense loss of sigmoid
        # output_dense = output_dense * 2 - 1.
        # output_dense = cre_dense(output_dense.reshape(output_dense.shape[0], -1))
        # loss_dense = torch.mul(torch.log(output_dense), target_dense) +\
        #              torch.mul(torch.log(1. - output_dense), (1 - target_dense))
        # loss_dense = loss_dense.mean() * (-10)

        loss = loss_moco + loss_dense * .2

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_moco, target_moco, topk=(1, 5))
        # acc_dense = torch.mul(nn.functional.softmax(output_dense, dim=1).reshape(output_dense.shape[0], -1),
        #                       target_dense).sum(dim=1).mean()
        acc_dense_pos = output_dense.reshape(output_dense.shape[0], -1).argmax(dim=1)
        acc_dense = target_dense[torch.arange(0, target_dense.shape[0]), acc_dense_pos].float().mean() * 100
        loss_m.update(loss_moco.item(), images[0].size(0))
        loss_s.update(loss_dense.item(), images[0].size(0))
        acc_moco.update(acc1[0], images[0].size(0))
        acc_seg.update(acc_dense.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if i % args.scalar_freq == 0 and torch.distributed.get_rank() == 0:
            global_step = i + epoch * (args.num_images // args.batch_size) / 4
            writer.add_scalar('loss_moco', loss_moco.item(), global_step)
            writer.add_scalar('loss_seg', loss_dense.item(), global_step)
            writer.add_scalar('acc1', acc1[0], global_step)
            writer.add_scalar('acc5', acc5[0], global_step)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        if torch.distributed.get_rank() == 0:
            logger_moco.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()