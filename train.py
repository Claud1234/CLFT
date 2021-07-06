#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import argparse
from tqdm import tqdm
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import configs
from fcn.dataloader import Dataset
from fcn.fusion_net import FusionNet
from utils.helpers import adjust_learning_rate
from utils.helpers import save_model_dict
from utils.metrics import find_overlap

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-r', '--resume_training', required=True,
                    dest='resume_training', choices=['yes', 'no'],
                    help='Training resuming or starting from the beginning')
parser.add_argument('-p', '--model_path', dest='model_path',
                    help='path of checkpoint for training resuming')
args = parser.parse_args()

# if os.path.exists('runs'):
#     shutil.rmtree('runs')

device = torch.device(configs.DEVICE)
writer = SummaryWriter()


def main():
    # Define the model
    model = FusionNet()
    model.to(device)
    print("Use Device: {} for training".format(configs.DEVICE))

    # Define loss function (criterion) and optimizer
    weight_loss = torch.Tensor(configs.CLASS_TOTAL).fill_(0)
    weight_loss[0] = 1
    weight_loss[1] = 3
    weight_loss[2] = 10
    criterion = nn.CrossEntropyLoss(weight=weight_loss).to(device)
    print('Criterion Initialization Succeed')
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)
    print('Optimizer Initialization Succeed')

    if args.resume_training == 'yes':
        print('Resume Training')
        checkpoint = torch.load(args.model_path)
        finsihed_epochs = checkpoint['epoch']
        print(f"Finsihed epochs in previous training: {finsihed_epochs}")
        if configs.EPOCHS <= finsihed_epochs:
            print('Present epochs amount is smaller than finished epochs!!!')
            print(f"Please setting the epochs bigger than {finsihed_epochs}")
            sys.exit()
        elif configs.EPOCHS > finsihed_epochs:
            print('Loading trained model weights...')
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Loading trained optimizer...')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif args.resume_training == 'no':
        print('Training from the beginning')
        finsihed_epochs = 0

    train_dataset = Dataset(dataroot=configs.DATAROOT,
                            split=configs.TRAIN_SPLITS,
                            augment=configs.AUGMENT)
    train_loader = DataLoader(train_dataset,
                              batch_size=configs.BATCH_SIZE,
                              num_workers=configs.WORKERS,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    valid_dataset = Dataset(dataroot=configs.DATAROOT,
                            split=configs.EVAL_SPLITS,
                            augment=None)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=configs.BATCH_SIZE,
                              num_workers=configs.WORKERS,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)

    for epoch in range(finsihed_epochs, configs.EPOCHS):
        curr_lr = adjust_learning_rate(optimizer, epoch, configs.EPOCHS)
        # One epoch training
        train_epoch_loss_rgb, train_epoch_IoU = train(
                                            train_dataset=train_dataset,
                                            train_loader=train_loader,
                                            model=model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch, lr=curr_lr)
        # One epoch validating
        valid_epoch_loss_rgb, valid_epoch_IoU = validate(
                                            valid_dataset=valid_dataset,
                                            valid_loader=valid_loader,
                                            model=model, criterion=criterion,
                                            epoch=epoch)
        # Plot the train and validation loss in Tensorboard
        writer.add_scalars('Loss', {'train': train_epoch_loss_rgb,
                                    'valid': valid_epoch_loss_rgb}, epoch)
        # Plot the train and validation IoU in Tensorboard
        writer.add_scalars('Background_IoU',
                           {'train': train_epoch_IoU[0],
                            'valid': valid_epoch_IoU[0]}, epoch)
        writer.add_scalars('Vehicle_IoU',
                           {'train': train_epoch_IoU[1],
                            'valid': valid_epoch_IoU[1]}, epoch)
        writer.add_scalars('Human_IoU',
                           {'train': train_epoch_IoU[2],
                            'valid': valid_epoch_IoU[2]}, epoch)
        writer.close()
        # Save the checkpoint
        if (epoch+1) % configs.SAVE_EPOCH == 0 and epoch > 0:
            print('Saving Model...')
            save_model_dict(epoch, model, optimizer)
            print('Saving Model Complete')
    print('Training Complete')


def train(train_dataset, train_loader, model, criterion, optimizer, epoch, lr):
    '''
    The training of one epoch
    '''
    model.train()
    print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
    print('Training...')
    train_loss_rgb = 0.0
    overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
    batches_amount = int(len(train_dataset)/configs.BATCH_SIZE)
    progress_bar = tqdm(train_loader, total=batches_amount)
    count = 0
    for _, batch in enumerate(progress_bar):
        count += 1
        batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
        batch['lidar'] = batch['lidar'].to(device, non_blocking=True)
        batch['annotation'] = \
            batch['annotation'].to(device, non_blocking=True).squeeze(1)

        optimizer.zero_grad()
        outputs = model(batch['rgb'], batch['lidar'], 'all')

        output = outputs['fusion']
        annotation = batch['annotation']
        batch_overlap, batch_pred, batch_label, batch_union = \
            find_overlap(output, annotation)

        overlap_cum += batch_overlap
        pred_cum += batch_pred
        label_cum += batch_label
        union_cum += batch_union

        loss_rgb = criterion(outputs['rgb'], batch['annotation'])
        loss_lidar = criterion(outputs['lidar'], batch['annotation'])
        loss_fusion = criterion(outputs['fusion'], batch['annotation'])
        loss = loss_rgb + loss_lidar + loss_fusion

        train_loss_rgb += loss_rgb.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_description(f'rgb loss:{loss_rgb:.4f}, ' +
                                     f'lidar loss:{loss_lidar:.4f}, ' +
                                     f'fusion loss:{loss_fusion:.4f},')
    # The IoU of one epoch
    train_epoch_IoU = overlap_cum / union_cum
    print(f'Training IoU of background for Epoch: {train_epoch_IoU[0]:.4f}')
    print(f'Training IoU of vehicles for Epoch: {train_epoch_IoU[1]:.4f}')
    print(f'Training IoU of human for Epoch: {train_epoch_IoU[2]:.4f}')
    # The loss_rgb of one epoch
    train_epoch_loss_rgb = train_loss_rgb / count
    print(f'Average Training RGB Loss for Epoch: {train_epoch_loss_rgb:.4f}')

    return train_epoch_loss_rgb, train_epoch_IoU


def validate(valid_dataset, valid_loader, model, criterion, epoch):
    '''
    The validation of one epoch
    '''
    model.eval()
    print('Validating...')
    valid_loss_rgb = 0.0
    overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
    with torch.no_grad():
        batches_amount = int(len(valid_dataset)/configs.BATCH_SIZE)
        progress_bar = tqdm(valid_loader, total=batches_amount)
        count = 0
        for _, batch in enumerate(progress_bar):
            count += 1
            batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
            batch['lidar'] = batch['lidar'].to(device, non_blocking=True)
            batch['annotation'] = \
                batch['annotation'].to(device, non_blocking=True).squeeze(1)

            outputs = model(batch['rgb'], batch['lidar'], 'all')

            output = outputs['fusion']
            annotation = batch['annotation']
            batch_overlap, batch_pred, batch_label, batch_union = \
                find_overlap(output, annotation)

            overlap_cum += batch_overlap
            pred_cum += batch_pred
            label_cum += batch_label
            union_cum += batch_union

            loss_rgb = criterion(outputs['rgb'], batch['annotation'])
            loss_lidar = criterion(outputs['lidar'], batch['annotation'])
            loss_fusion = criterion(outputs['fusion'], batch['annotation'])
            # loss = loss_rgb + loss_lidar + loss_fusion

            valid_loss_rgb += loss_rgb.item()

            progress_bar.set_description(f'rgb loss:{loss_rgb:.4f}, ' +
                                         f'lidar loss:{loss_lidar:.4f}, ' +
                                         f'fusion loss:{loss_fusion:.4f},')
    # The IoU of one epoch
    valid_epoch_IoU = overlap_cum / union_cum
    print(f'Validatoin IoU of background for Epoch: {valid_epoch_IoU[0]:.4f}')
    print(f'Validatoin IoU of vehicles for Epoch: {valid_epoch_IoU[1]:.4f}')
    print(f'Validatoin IoU of human for Epoch: {valid_epoch_IoU[2]:.4f}')
    # The loss_rgb of one epoch
    valid_epoch_loss_rgb = valid_loss_rgb / count
    print(f'Average Validation RGB Loss for Epoch: {valid_epoch_loss_rgb:.4f}')

    return valid_epoch_loss_rgb, valid_epoch_IoU


if __name__ == '__main__':
    main()
