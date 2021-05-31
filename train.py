#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from tqdm import tqdm
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

import configs
from fcn.dataloader import Dataset
from fcn.fusion_net import FusionNet
from utils.helpers import adjust_learning_rate
from utils.helpers import save_model_dict


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-r', '--resume_training', required=True,
                    dest='resume_training', choices=['yes', 'no'],
                    help='Training resuming or starting from the beginning')
parser.add_argument('-p', '--model_path', dest='model_path',
                    help='path of checkpoint for training resuming')
args = parser.parse_args()

# Define the model
model = FusionNet()
device = torch.device(configs.DEVICE)
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
                        split=configs.SPLITS,
                        augment=configs.AUGMENT)
train_loader = DataLoader(train_dataset,
                          batch_size=configs.BATCH_SIZE,
                          num_workers=configs.WORKERS,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

for epoch in range(finsihed_epochs, configs.EPOCHS):
    curr_lr = adjust_learning_rate(optimizer, epoch, configs.EPOCHS)
    model.train()
    print('Training')
    print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, curr_lr))
    batches_amount = int(len(train_dataset)/configs.BATCH_SIZE)
    progress_bar = tqdm(train_loader, total=batches_amount)
    for i, batch in enumerate(progress_bar):
        batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
        batch['lidar'] = batch['lidar'].to(device, non_blocking=True)
        batch['annotation'] = \
            batch['annotation'].to(device, non_blocking=True).squeeze(1)

        optimizer.zero_grad()
        output = model(batch['rgb'], batch['lidar'], 'all')

        loss_rgb = criterion(output['rgb'], batch['annotation'])
        loss_lidar = criterion(output['lidar'], batch['annotation'])
        loss_fusion = criterion(output['fusion'], batch['annotation'])
        loss = loss_rgb + loss_lidar + loss_fusion

        loss.backward()
        optimizer.step()

        progress_bar.set_description(desc=f'rgb loss: {loss_rgb:.4f}' '|'
                                          f'lidar_loss: {loss_lidar:.4f}' '|'
                                          f'fusion_loss: {loss_fusion:.4f}' '|'
                                          f'loss: {loss:.4f}')

    if (epoch+1) % configs.SAVE_EPOCH == 0 and epoch > 0:
        print('Saving Model...')
        save_model_dict(epoch, model, optimizer)
        print('Saving Model Complete')

print('Training Complete')
