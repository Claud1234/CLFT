#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np


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
import torch.nn.functional as F
import torchvision

import configs
from fcn.dataloader import Dataset
from fcn.fusion_net import FusionNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR', #help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochs_cotrain', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--n_classes', default=4, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lrsemi', default=0.00005, type=float, metavar='N',
                    help='number of classes')
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
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch'
                         'N processes per node, which has N GPUs. This is the'
                         'fastest way to use PyTorch for either single node or'
                         'multi node data parallel training')

logdir = configs.LOG_DIR
if not os.path.exists(logdir):
    os.makedirs(logdir)


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

    ngpus_per_node = torch.cuda.device_count()  # 1
    print(args.multiprocessing_distributed)
    if args.multiprocessing_distributed:

        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print('multi')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,
                                                           args))
    else:
        # Simply call main_worker function
        print('single')
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print(args.gpu)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

#     if args.distributed:
#         print(args.distributed)
#         print('dist_url', args.dist_url)
#         if args.dist_url == "env://" and args.rank == -1:
#             args.rank = int(os.environ["RANK"])
#         if args.multiprocessing_distributed:
#             # For multiprocessing distributed training, rank needs to be the
#             # global rank among all the processes
#             args.rank = args.rank * ngpus_per_node + gpu
#         dist.init_process_group(backend=args.dist_backend,
#                                 init_method=args.dist_url,
#                                 world_size=args.world_size, rank=args.rank)
    # create model
    model = FusionNet()

#     if args.distributed:
#         print('distributed', args.distributed)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
#         args.batch_size = int(args.batch_size / ngpus_per_node)
#         args.workers = int((args.workers + ngpus_per_node - 1) /
#                            ngpus_per_node)
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, device_ids=[args.gpu], find_unused_parameters=True)

    # define loss function (criterion) and optimizer
    weight_loss = torch.Tensor(args.n_classes).fill_(0)
    weight_loss[0] = 1
    weight_loss[1] = 3
    weight_loss[2] = 10
    criterion = nn.CrossEntropyLoss(weight=weight_loss).cuda(args.gpu)
    # criterion = nn.CrossEntropyLoss(weight=weight_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))

    cudnn.benchmark = True
    save_epoch = 10
    train_dataset = Dataset(dataroot=configs.DATAROOT,
                            split=configs.SPLITS, augment=configs.AUGMENT)

    semi_dataset = Dataset(dataroot=configs.DATAROOT,
                           split=configs.SPLITS, augment=configs.AUGMENT)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                                train_dataset)
        semi_sampler = torch.utils.data.distributed.DistributedSampler(
                                                                semi_dataset)
    else:
        train_sampler = None
        semi_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=configs.WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    semi_loader = torch.utils.data.DataLoader(
        semi_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=(semi_sampler is None),
        num_workers=configs.WORKERS,
        pin_memory=True,
        sampler=semi_sampler,
        drop_last=True)

    '''
    #### Resume and Evalutation
    '''
    # loc = 'cuda:{}'.format(args.gpu)
    # cpu_loc = 'cpu'
    # print (model)
    # checkpoint = torch.load(
         # '/home/claude/Data/logs/2nd_test/checkpoint_0009.pth.tar', 
                                                        # map_location=cpu_loc)
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # evaluation(train_loader, model, criterion, optimizer, args)
    #
    # del checkpoint
    # #torch.cuda.empty_cache()

    '''
    #### Super
    '''
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            semi_sampler.set_epoch(epoch)

        curr_lr = adjust_learning_rate(optimizer, epoch, args.epochs, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, args)
        print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, curr_lr))
        if (epoch+1) % save_epoch == 0 and epoch > 0:
            if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and
                 args.rank % ngpus_per_node == 0):
                save_checkpoint(
                    {'epoch': epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), },
                    is_best=False,
                    filename=logdir + 'checkpoint_{:04d}.pth'.format(epoch))

    '''
    #### Co-training
    '''
    # for epoch in range(args.start_epoch, args.epochs_cotrain):
        # if args.distributed:
            # train_sampler.set_epoch(epoch)
            # semi_sampler.set_epoch(epoch)
            #
        # curr_lr = adjust_learning_rate_semi(optimizer, epoch, 
                                            # args.epochs_cotrain, args)
                                            #
        # # train for one epoch
        # train_semi(train_loader, semi_loader, model, criterion, optimizer, args)   
        # print('Semi -- Epoch: {:.0f}, LR: {:.6f}'.format(epoch, curr_lr))
        # if (epoch+1) % save_epoch == 0 and epoch > 0:
            # if not args.multiprocessing_distributed or \
            # (args.multiprocessing_distributed and 
             # args.rank % ngpus_per_node == 0):
                # save_checkpoint(
                    # {'epoch': epoch + 1,
                    # 'state_dict': model.state_dict(),
                    # 'optimizer' : optimizer.state_dict(),}, 
                    # is_best=False, 
                    # filename=logdir+'checkpoint_cotrain_{:04d}.pth.tar'.format(
                                                                        # epoch))


def train(train_loader, model, criterion, optimizer, args):
    # switch to train mode
    print('full _train')
    model.train()

    for batch in train_loader:
        # measure data loading time
        if args.gpu is not None:
            batch['rgb'] = batch['rgb'].cuda(args.gpu, non_blocking=True)
            batch['lidar'] = batch['lidar'].cuda(args.gpu, non_blocking=True)
            batch['annotation'] = \
                batch['annotation'].cuda(args.gpu, non_blocking=True).squeeze(1)

        # compute output
        output = model(batch['rgb'], batch['lidar'], 'all')

        loss_rgb = criterion(output['rgb'], batch['annotation'])
        loss_lidar = criterion(output['lidar'], batch['annotation'])
        loss_fusion = criterion(output['fusion'], batch['annotation'])
        loss = loss_rgb+loss_lidar+loss_fusion

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_semi(train_loader, semi_loader, model, criterion, optimizer, args):
    lambda_cot = 1
    print('semi_train')
    model.train()
    for batch in train_loader:
        # measure data loading time
        if args.gpu is not None:
            batch['rgb'] = batch['rgb'].cuda(args.gpu, non_blocking=True)
            batch['lidar'] = batch['lidar'].cuda(args.gpu, non_blocking=True)
            batch['annotation'] = batch['annotation'].cuda(
                                        args.gpu, non_blocking=True).squeeze(1)

        output = model(batch['rgb'], batch['lidar'], 'all')

        loss_rgb = criterion(output['rgb'], batch['annotation'])
        loss_lidar = criterion(output['lidar'], batch['annotation'])
        loss_fusion = criterion(output['fusion'], batch['annotation'])
        loss = loss_rgb + loss_lidar + loss_fusion

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:', loss)

        try:
            batch = next(unsuper_dataloader)[1]
        except:
            unsuper_dataloader = enumerate(semi_loader)
            batch = next(unsuper_dataloader)[1]

        if args.gpu is not None:
            batch['rgb'] = batch['rgb'].cuda(args.gpu, non_blocking=True)
            batch['lidar'] = batch['lidar'].cuda(args.gpu, non_blocking=True)
            batch['annotation'] = batch['annotation'].cuda(
                                args.gpu, non_blocking=True).squeeze(1)

        with torch.no_grad():
            model.eval()
            output = model(batch['rgb'], batch['lidar'], 'fusion')
            annotation_teacher = F.softmax(output['fusion'], 1)
            _, annotation_teacher = torch.max(annotation_teacher, 1)
            mask_not_valid = batch['annotation'] == 3
            annotation_teacher[mask_not_valid] = 3

        model.train()
        output = model(batch['rgb'], batch['lidar'], 'ind')
        loss_rgb = lambda_cot*criterion(output['rgb'],
                                        annotation_teacher.detach().clone())
        loss_lidar = lambda_cot*criterion(output['lidar'],
                                          annotation_teacher.detach().clone())
        loss_unsuper = loss_rgb + loss_lidar

        # output = model(batch['rgb'], batch['lidar'],'all')

        # loss_rgb = criterion(output['rgb'], batch['annotation'])
        # loss_lidar = criterion(output['lidar'], batch['annotation'])
        # loss_fusion = criterion(output['fusion'], batch['annotation'])
        # loss_unsuper = loss_rgb+loss_lidar+loss_fusion

        optimizer.zero_grad()
        loss_unsuper.backward()
        optimizer.step()


def evaluation(train_dataset, model, criterion, optimizer, args):
    model.eval()
    print('evaluation')
    with torch.no_grad():
        for batch in train_dataset:
            output = model(batch['rgb'], batch['lidar'], 'ind')
        # annotation_teacher = F.softmax(output['fusion'], 1)
        # _, annotation_teacher = torch.max(annotation_teacher, 1)
        # mask_not_valid = batch['annotation'] == 3
        # annotation_teacher[mask_not_valid] = 3
        #print('output:', output)
        #print(output['rgb'].shape)
        # create a color pallette, selecting a color for each class
        # from PIL import Image
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")
        #
        # # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(output['rgb'].byte().cpu().numpy()).resize(rgb.size)
        # r.putpalette(colors)
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(r)
        # plt.show()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, epoch_max, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr * (1 - epoch/epoch_max)**0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_semi(optimizer, epoch, epoch_max, args):
    mid_epoch = epoch_max/2
    if epoch <= mid_epoch:
        lr = np.exp(-(1-epoch/mid_epoch)**2)*args.lrsemi
    else:
        lr = args.lrsemi * (1 - epoch/epoch_max)**0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
