#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Semi-supervised training script

Created on Feb. th, 2022
'''
import sys
import argparse
from tqdm import tqdm
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import configs
from fcn.dataloader import Dataset
from fcn.dataloader import SemiDataset
from fcn.fusion_net import FusionNet
from utils.helpers import adjust_learning_rate_semi
from utils.helpers import save_model_dict
from utils.helpers import EarlyStopping
from utils.metrics import find_overlap

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('-r', '--resume_training', required=True,
#                     dest='resume_training', choices=['yes', 'no'],
#                     help='Training resuming or starting from the beginning')
# parser.add_argument('-reset-lr', dest='reset_lr', action='store_true',
#                     help='Reset LR to initial value defined in configs')
parser.add_argument('-p', '--model_path', dest='model_path', required=True,
                    help='path of checkpoint for training resuming')
# parser.add_argument('-i', '--dataset', dest='dataset', type=str, required=True,
#                     help='select to evaluate waymo or iseauto dataset')
parser.add_argument('-m', '--model', dest='model', required=True,
                    choices=['rgb', 'lidar', 'fusion'],
                    help='Define training modes. (rgb, lidar or fusion)')
args = parser.parse_args()

device = torch.device(configs.DEVICE)
writer = SummaryWriter()


def main():
    # Define the model
    model = FusionNet()
    model.to(device)
    print("Use Device: {} for training".format(configs.DEVICE))

    early_stopping = EarlyStopping()

    # Define loss function (criterion) and optimizer
    weight_loss = torch.Tensor(configs.CLASS_TOTAL).fill_(0)
#    weight_loss[3] = 1
    weight_loss[0] = 1
    weight_loss[1] = 3
    weight_loss[2] = 10
    criterion = nn.CrossEntropyLoss(weight=weight_loss).to(device)
    print('Criterion Initialization Succeed')
    if args.model == 'rgb':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs.LR_SEMI_RGB)
    elif args.model == 'lidar':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs.LR_SEMI_LIDAR)
    elif args.model == 'fusion':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs.LR_SEMI_FUSION)
    else:
        sys.exit("You have to specify a training mode.(rgb, lidar or fusion)")
    print('Optimizer Initialization Succeed')

#     if args.resume_training == 'yes':
#         print('Resume Training')
    checkpoint = torch.load(args.model_path)
#         if args.reset_lr is True:
#             print('Reset the epoch to 0')
#             finsihed_epochs = 0
#         else:
#             finsihed_epochs = checkpoint['epoch']
#             print(f"Finsihed epochs in previous training: {finsihed_epochs}")
#         if configs.EPOCHS <= finsihed_epochs:
#             print('Present epochs amount is smaller than finished epochs!!!')
#             print(f"Please setting the epochs bigger than {finsihed_epochs}")
#             sys.exit()
#         elif configs.EPOCHS > finsihed_epochs:
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loading trained optimizer...')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     elif args.resume_training == 'no':
#         print('Training from the beginning')
#         finsihed_epochs = 0

#     if args.dataset == 'waymo':
#         train_dataset = Dataset(dataset=args.dataset,
#                                 rootpath=configs.WAY_ROOTPATH,
#                                 split=configs.WAY_TRAIN_SPLITS,
#                                 augment=True)
#     elif args.dataset == 'iseauto':
    semi_train_dataset = SemiDataset(rootpath=configs.ISE_ROOTPATH,
                                     split=configs.ISE_SEMI_TRAIN_SPLITS,
                                     augment=True)
    semi_train_loader = DataLoader(semi_train_dataset,
                                   batch_size=configs.BATCH_SIZE,
                                   num_workers=configs.WORKERS,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True)

#     if args.dataset == 'waymo':
#         valid_dataset = Dataset(dataset=args.dataset,
#                                 rootpath=configs.WAY_ROOTPATH,
#                                 split=configs.WAY_VALID_SPLITS,
#                                 augment=None)
#     elif args.dataset == 'iseauto':
    valid_dataset = Dataset(dataset='iseauto',
                            rootpath=configs.ISE_ROOTPATH,
                            split=configs.ISE_VALID_SPLITS,
                            augment=None)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=configs.BATCH_SIZE,
                              num_workers=configs.WORKERS,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)

    for epoch in range(0, configs.EPOCHS_SEMI):
        curr_lr = adjust_learning_rate_semi(args.model, optimizer,
                                            epoch, configs.EPOCHS_SEMI)
        # One epoch training
        train_epoch_loss, train_epoch_IoU = semi_train(
                                            train_dataset=semi_train_dataset,
                                            train_loader=semi_train_loader,
                                            model=model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch, lr=curr_lr)
        # One epoch validating
        valid_epoch_loss, valid_epoch_IoU = validate(
                                            valid_dataset=valid_dataset,
                                            valid_loader=valid_loader,
                                            model=model, criterion=criterion,
                                            epoch=epoch)
        # Plot the train and validation loss in Tensorboard
        writer.add_scalars('Loss', {'train': train_epoch_loss,
                                    'valid': valid_epoch_loss}, epoch)
        # Plot the train and validation IoU in Tensorboard
        writer.add_scalars('Vehicle_IoU',
                           {'train': train_epoch_IoU[0],
                            'valid': valid_epoch_IoU[0]}, epoch)
        writer.add_scalars('Human_IoU',
                           {'train': train_epoch_IoU[1],
                            'valid': valid_epoch_IoU[1]}, epoch)
        writer.close()

        # Save the checkpoint
        if configs.EARLY_STOPPING is True:
            early_stopping(valid_epoch_loss, epoch, model, optimizer)
            if early_stopping.early_stop_trigger is True:
                break
        else:
            if (epoch+1) % configs.SAVE_EPOCH == 0 and epoch > 0:
                print('Saving Model...')
                save_model_dict(epoch, model, optimizer)
                print('Saving Model Complete')
    print('Training Complete')


def semi_train(train_dataset, train_loader, model,
               criterion, optimizer, epoch, lr):
    '''
    The semi-training of one epoch
    '''
    print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
    print('Semi-supervised training...')
    train_loss = 0.0
    overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
    batches_amount = int(len(train_dataset)/configs.BATCH_SIZE)
    progress_bar = tqdm(train_loader, total=batches_amount)
    count = 0
    for _, batch in enumerate(progress_bar):
        count += 1
        batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
        batch['lidar'] = batch['lidar'].to(device, non_blocking=True)

        with torch.no_grad():
            model.eval()
            outputs = model(batch['rgb'], batch['lidar'], 'all')
            output = outputs[args.model]

            annotation_teacher = nn.functional.softmax(output, 1)
            _, annotation_teacher = torch.max(annotation_teacher, 1)

        model.train()
        optimizer.zero_grad()
        outputs = model(batch['rgb'], batch['lidar'], 'all')

        output = outputs[args.model]
        annotation = annotation_teacher.detach().clone()
        batch_overlap, batch_pred, batch_label, batch_union = \
            find_overlap(output, annotation)

        overlap_cum += batch_overlap
        pred_cum += batch_pred
        label_cum += batch_label
        union_cum += batch_union

        loss_rgb = criterion(outputs['rgb'], annotation)
        loss_lidar = criterion(outputs['lidar'], annotation)
        loss_fusion = criterion(outputs['fusion'], annotation)
        loss = loss_rgb + loss_lidar + loss_fusion

        if args.model == 'rgb':
            train_loss += loss_rgb.item()
            loss_rgb.backward()
            optimizer.step()
            progress_bar.set_description(f'semi-train rgb loss:{loss_rgb:.4f}')

        elif args.model == 'lidar':
            train_loss += loss_lidar.item()
            loss_lidar.backward()
            optimizer.step()
            progress_bar.set_description(
                f'semi-train lidar loss:{loss_lidar:.4f}')

        elif args.model == 'fusion':
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'semi-train fusion loss:{loss:.4f}')
    # The IoU of one epoch
    train_epoch_IoU = overlap_cum / union_cum
    print(f'Semi-training IoU of vehicles for Epoch: {train_epoch_IoU[0]:.4f}')
    print(f'Semi-training IoU of human for Epoch: {train_epoch_IoU[1]:.4f}')
    # The loss_rgb of one epoch
    train_epoch_loss = train_loss / count
    print(f'Average Semi-training Loss for Epoch: {train_epoch_loss:.4f}')

    return train_epoch_loss, train_epoch_IoU


def validate(valid_dataset, valid_loader, model, criterion, epoch):
    '''
    The validation of one epoch
    '''
    model.eval()
    print('Validating...')
    valid_loss = 0.0
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

            output = outputs[args.model]
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

            if args.model == 'rgb':
                valid_loss += loss_rgb.item()
                progress_bar.set_description(f'valid rgb loss:{loss_rgb:.4f}')

            elif args.model == 'lidar':
                valid_loss += loss_lidar.item()
                progress_bar.set_description(
                    f'valid lidar loss:{loss_lidar:.4f}')

            elif args.model == 'fusion':
                valid_loss += loss.item()
                progress_bar.set_description(f'valid fusion loss:{loss:.4f}')
    # The IoU of one epoch
    valid_epoch_IoU = overlap_cum / union_cum
    print(f'Validatoin IoU of vehicles for Epoch: {valid_epoch_IoU[0]:.4f}')
    print(f'Validatoin IoU of human for Epoch: {valid_epoch_IoU[1]:.4f}')
    # The loss_rgb of one epoch
    valid_epoch_loss = valid_loss / count
    print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

    return valid_epoch_loss, valid_epoch_IoU


if __name__ == '__main__':
    main()
