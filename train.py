#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import argparse
from tqdm import tqdm
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import configs
from iseauto.trainer import Trainer
from iseauto.dataset import Dataset
from fcn.fusion_net import FusionNet
from utils.helpers import adjust_learning_rate
from utils.helpers import save_model_dict
from utils.helpers import EarlyStopping
from utils.metrics import find_overlap


with open('config.json', 'r') as f:
    config = json.load(f)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-bb', '--backbone', required=True,
                    choices=['fcn', 'dpt'],
                    help='Use the backbone of training, dpt or fcn')
parser.add_argument('-reset-lr', dest='reset_lr', action='store_true',
                    help='Reset LR to initial value defined in configs')
parser.add_argument('-p', '--model_path', dest='model_path',
                    help='path of checkpoint for training resuming')
parser.add_argument('-i', '--dataset', dest='dataset', type=str, required=True,
                    help='select to evaluate waymo or iseauto dataset')
parser.add_argument('-m', '--model', dest='model', required=True,
                    choices=['rgb', 'lidar', 'fusion'],
                    help='Define training modes. (rgb, lidar or fusion)')
args = parser.parse_args()

trainer = Trainer(config, args)
#Trainer.train()

writer = SummaryWriter()



# if os.path.exists('runs'):
#     shutil.rmtree('runs')



def main():

    # Define loss function (criterion) and optimizer
    weight_loss = torch.Tensor(configs.CLASS_TOTAL).fill_(0)
#    weight_loss[3] = 1
    weight_loss[0] = 1
    weight_loss[1] = 3
    weight_loss[2] = 10
    criterion = nn.CrossEntropyLoss(weight=weight_loss).to(device)
    print('Criterion Initialization Succeed')
    if args.model == 'rgb':
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR_RGB)
    elif args.model == 'lidar':
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR_LIDAR)
    elif args.model == 'fusion':
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR_FUSION)
    else:
        sys.exit("You have to specify a training mode.(rgb, lidar or fusion)")
    print('Optimizer Initialization Succeed')

    if config['General']['resume_training']:
        resume_train_model = config['General']['resume_training_model_path']
        print(f'Resume Training from {resume_train_model}')
        checkpoint = torch.load(resume_train_model)
        if args.reset_lr is True:
            print('Reset the epoch to 0')
            finsihed_epochs = 0
        else:
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
    else:
        print('Training from the beginning')
        finsihed_epochs = 0

    if args.dataset == 'waymo':
        train_dataset = Dataset(dataset=args.dataset,
                                rootpath=configs.WAY_ROOTPATH,
                                split=configs.WAY_TRAIN_SPLITS,
                                augment=True)
    elif args.dataset == 'iseauto':
        train_dataset = Dataset(dataset=args.dataset,
                                rootpath=configs.ISE_ROOTPATH,
                                split=configs.ISE_TRAIN_SPLITS,
                                augment=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=configs.BATCH_SIZE,
                              num_workers=configs.WORKERS,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    if args.dataset == 'waymo':
        valid_dataset = Dataset(dataset=args.dataset,
                                rootpath=configs.WAY_ROOTPATH,
                                split=configs.WAY_VALID_SPLITS,
                                augment=None)
    elif args.dataset == 'iseauto':
        valid_dataset = Dataset(dataset=args.dataset,
                                rootpath=configs.ISE_ROOTPATH,
                                split=configs.ISE_VALID_SPLITS,
                                augment=None)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=configs.BATCH_SIZE,
                              num_workers=configs.WORKERS,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)

    for epoch in range(finsihed_epochs, configs.EPOCHS):
        curr_lr = adjust_learning_rate(args.model, optimizer,
                                       epoch, configs.EPOCHS)
        # One epoch training
        train_epoch_loss, train_epoch_IoU = train(
                                            train_dataset=train_dataset,
                                            train_loader=train_loader,
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
        early_stopping(args.model, valid_epoch_loss, epoch, model, optimizer)
        if (epoch+1) % configs.SAVE_EPOCH == 0 and epoch > 0:
            print('Saving model for every 10 epochs...')
            save_model_dict(args.model, epoch, model, optimizer, True)
            print('Saving Model Complete')
        if early_stopping.early_stop_trigger is True:
                break

    print('Training Complete')





if __name__ == '__main__':
    main()
