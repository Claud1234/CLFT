#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fcn.fusion_net import FusionNet

#from FOD.utils import get_losses, get_optimizer, get_schedulers, create_dir
from dpt.dpt import DPT
from utils.helpers import EarlyStopping


class Trainer(object):
	def __init__(self, config, args):
		super().__init__()
		self.config = config
		self.args = args
		self.type = self.config['General']['type']

		self.device = torch.device(self.config['General']['device']
								   if torch.cuda.is_available() else "cpu")
		print("device: %s" % self.device)

		if args.backbone == 'fcn':
			self.model = FusionNet()
			print(f'Using backbone {args.backbone}')
			self.optimizer_fcn = torch.optim.Adam(self.model.parameters(),
							lr = config['General']['fcn'][
								f"lr_{config['General']['sensor_modality']}"])

		elif args.backbone == 'dpt':
			resize = config['Dataset']['transforms']['resize']
			self.model = DPT(
				image_size=(3, resize, resize),
				emb_dim=config['General']['emb_dim'],
				resample_dim=config['General']['resample_dim'],
				read=config['General']['read'],
				nclasses=len(config['Dataset']['classes']) + 1,
				hooks=config['General']['hooks'],
				model_timm=config['General']['model_timm'],
				type=self.type,
				patch_size=config['General']['patch_size'], )
			print(f'Using backbone {args.backbone}')
			self.optimizer_dpt_backbone, self.optimizer_dpt_scratch = \
				get_optimizer_dpt(config, self.model)
			self.scheduler_backbone = ReduceLROnPlateau(self.optimizer_dpt_backbone)
			self.scheduler_scratch = ReduceLROnPlateau(self.optimizer_dpt_scratch)

		else:
			sys.exit("A backbone must be specified! (dpt or fcn)")

		self.model.to(self.device)

		self.early_stopping = EarlyStopping()
		nclasses = len(config['Dataset']['classes'])
		weight_loss = torch.Tensor(nclasses).fill_(0)
		# define weight of different classes, 0-background, 1-car, 2-people.
		# weight_loss[3] = 1
		weight_loss[0] = 1
		weight_loss[1] = 3
		weight_loss[2] = 10
		self.criterion = nn.CrossEntropyLoss(weight=weight_loss).to(self.device)

	def train(self, train_dataset, train_loader, model, criterion, optimizer,
			  epoch, lr):
		"""
		The training of one epoch
		"""
		epochs = self.config['General']['epochs']
		batch_size = self.config['General']['batch_size']
		self.model.train()
		print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
		print('Training...')
		train_loss = 0.0
		overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
		batches_amount = int(len(train_dataset) / batch_size)
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
				train_loss += loss_rgb.item()
				loss_rgb.backward()
				optimizer.step()
				progress_bar.set_description(f'train rgb loss:{loss_rgb:.4f}')

			elif args.model == 'lidar':
				train_loss += loss_lidar.item()
				loss_lidar.backward()
				optimizer.step()
				progress_bar.set_description(
					f'train lidar loss:{loss_lidar:.4f}')

			elif args.model == 'fusion':
				train_loss += loss.item()
				loss.backward()
				optimizer.step()
				progress_bar.set_description(f'train fusion loss:{loss:.4f}')
		# The IoU of one epoch
		train_epoch_IoU = overlap_cum / union_cum
		print(f'Training IoU of vehicles for Epoch: {train_epoch_IoU[0]:.4f}')
		print(f'Training IoU of human for Epoch: {train_epoch_IoU[1]:.4f}')
		# The loss_rgb of one epoch
		train_epoch_loss = train_loss / count
		print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

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
		batches_amount = int(len(valid_dataset) / configs.BATCH_SIZE)
		progress_bar = tqdm(valid_loader, total=batches_amount)
		count = 0
		for _, batch in enumerate(progress_bar):
			count += 1
			batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
			batch['lidar'] = batch['lidar'].to(device,
											   non_blocking=True)
			batch['annotation'] = \
				batch['annotation'].to(device,
									   non_blocking=True).squeeze(1)

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
			loss_lidar = criterion(outputs['lidar'],
								   batch['annotation'])
			loss_fusion = criterion(outputs['fusion'],
									batch['annotation'])
			loss = loss_rgb + loss_lidar + loss_fusion

			if args.model == 'rgb':
				valid_loss += loss_rgb.item()
				progress_bar.set_description(
					f'valid rgb loss:{loss_rgb:.4f}')

			elif args.model == 'lidar':
				valid_loss += loss_lidar.item()
				progress_bar.set_description(
					f'valid lidar loss:{loss_lidar:.4f}')

			elif args.model == 'fusion':
				valid_loss += loss.item()
				progress_bar.set_description(
					f'valid fusion loss:{loss:.4f}')
	# The IoU of one epoch
	valid_epoch_IoU = overlap_cum / union_cum
	print(
		f'Validatoin IoU of vehicles for Epoch: {valid_epoch_IoU[0]:.4f}')
	print(
		f'Validatoin IoU of human for Epoch: {valid_epoch_IoU[1]:.4f}')
	# The loss_rgb of one epoch
	valid_epoch_loss = valid_loss / count
	print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

	return valid_epoch_loss, valid_epoch_IoU


def create_dir(directory):
	try:
		os.makedirs(directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


# def get_optimizer(config, net):
#     if config['General']['optim'] == 'adam':
#         optimizer = optim.Adam(net.parameters(), lr=config['General']['lr'])
#     elif config['General']['optim'] == 'sgd':
#         optimizer = optim.SGD(net.parameters(), lr=config['General']['lr'], momentum=config['General']['momentum'])
#     return optimizer


def get_optimizer_fcn(config, net):
	return


def get_optimizer_dpt(config, net):
	names = set([name.split('.')[0] for name, _ in net.named_modules()]) - \
			{'', 'transformer_encoders'}
	params_backbone = net.transformer_encoders.parameters()
	params_scratch = list()
	for name in names:
		params_scratch += list(eval("net." + name).parameters())

	optimizer_dpt_backbone = torch.optim.Adam(params_backbone,
									lr=config['General']['dpt']['lr_backbone'])
	optimizer_dpt_scratch = torch.optim.Adam(params_scratch,
								   lr=config['General']['dpt']['lr_scratch'])

	return optimizer_dpt_backbone, optimizer_dpt_scratch


def get_schedulers(optimizers):
	return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]


def train(self, train_dataloader, val_dataloader):
	epochs = self.config['General']['epochs']
	if self.config['wandb']['enable']:
		wandb.init(project="FocusOnDepth",
				   entity=self.config['wandb']['username'])
		wandb.config = {
			"learning_rate_backbone": self.config['General']['lr_backbone'],
			"learning_rate_scratch": self.config['General']['lr_scratch'],
			"epochs": epochs,
			"batch_size": self.config['General']['batch_size']
		}
	val_loss = Inf
	for epoch in range(epochs):  # loop over the dataset multiple times
		print("Epoch ", epoch + 1)
		running_loss = 0.0
		self.model.train()
		pbar = tqdm(train_dataloader)
		pbar.set_description("Training")
		for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
			# get the inputs; data is a list of [inputs, labels]
			X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(
				self.device), Y_segmentations.to(self.device)
			# zero the parameter gradients
			self.optimizer_backbone.zero_grad()
			self.optimizer_scratch.zero_grad()
			# forward + backward + optimizer
			output_depths, output_segmentations = self.model(X)
			output_depths = output_depths.squeeze(
				1) if output_depths != None else None

			Y_depths = Y_depths.squeeze(1)  # 1xHxW -> HxW
			Y_segmentations = Y_segmentations.squeeze(1)  # 1xHxW -> HxW
			# get loss
			loss = self.loss_depth(output_depths,
								   Y_depths) + self.loss_segmentation(
				output_segmentations, Y_segmentations)
			loss.backward()
			# step optimizer
			self.optimizer_scratch.step()
			self.optimizer_backbone.step()

			running_loss += loss.item()
			if np.isnan(running_loss):
				print('\n',
					  X.min().item(), X.max().item(), '\n',
					  Y_depths.min().item(), Y_depths.max().item(), '\n',
					  output_depths.min().item(),
					  output_depths.max().item(), '\n',
					  loss.item(),
					  )
				exit(0)

			if self.config['wandb']['enable'] and (
					(i % 50 == 0 and i > 0) or i == len(
					train_dataloader) - 1):
				wandb.log({"loss": running_loss / (i + 1)})
			pbar.set_postfix({'training_loss': running_loss / (i + 1)})

		new_val_loss = self.run_eval(val_dataloader)

		if new_val_loss < val_loss:
			self.save_model(epoch)
			val_loss = new_val_loss

		self.schedulers[0].step(new_val_loss)
		self.schedulers[1].step(new_val_loss)

	print('Finished Training')

	def run_eval(self, val_dataloader):
		"""
			Evaluate the model on the validation set and visualize some results
			on wandb
			:- val_dataloader -: torch dataloader
		"""
		val_loss = 0.
		self.model.eval()
		X_1 = None
		Y_depths_1 = None
		Y_segmentations_1 = None
		output_depths_1 = None
		output_segmentations_1 = None
		with torch.no_grad():
			pbar = tqdm(val_dataloader)
			pbar.set_description("Validation")
			for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
				X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(
					self.device), Y_segmentations.to(self.device)
				output_depths, output_segmentations = self.model(X)
				output_depths = output_depths.squeeze(
					1) if output_depths != None else None
				Y_depths = Y_depths.squeeze(1)
				Y_segmentations = Y_segmentations.squeeze(1)
				if i == 0:
					X_1 = X
					Y_depths_1 = Y_depths
					Y_segmentations_1 = Y_segmentations
					output_depths_1 = output_depths
					output_segmentations_1 = output_segmentations
				# get loss
				loss = self.loss_depth(output_depths,
									   Y_depths) + self.loss_segmentation(
					output_segmentations, Y_segmentations)
				val_loss += loss.item()
				pbar.set_postfix({'validation_loss': val_loss / (i + 1)})
			if self.config['wandb']['enable']:
				wandb.log({"val_loss": val_loss / (i + 1)})
				self.img_logger(X_1, Y_depths_1, Y_segmentations_1,
								output_depths_1, output_segmentations_1)
		return val_loss / (i + 1)

	def save_model(self, epoch):
		path_model = os.path.join(self.config['General']['path_model'],
								  self.model.__class__.__name__)
		create_dir(path_model)
		print(path_model)
		torch.save({'model_state_dict': self.model.state_dict(),
					'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
					'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
					}, path_model + "/" + f"checkpoint_{epoch}.p")
		print('Model saved at : {}'.format(path_model))

	def img_logger(self, X, Y_depths, Y_segmentations, output_depths,
				   output_segmentations):
		nb_to_show = self.config['wandb']['images_to_show'] if \
		self.config['wandb']['images_to_show'] <= len(X) else len(X)
		tmp = X[:nb_to_show].detach().cpu().numpy()
		imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
		if output_depths != None:
			tmp = Y_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
			depth_truths = np.repeat(tmp, 3, axis=1)
			tmp = output_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
			tmp = np.repeat(tmp, 3, axis=1)
			# depth_preds = 1.0 - tmp
			depth_preds = tmp
		if output_segmentations != None:
			tmp = Y_segmentations[:nb_to_show].unsqueeze(
				1).detach().cpu().numpy()
			segmentation_truths = np.repeat(tmp, 3, axis=1).astype('float32')
			tmp = torch.argmax(output_segmentations[:nb_to_show], dim=1)
			tmp = tmp.unsqueeze(1).detach().cpu().numpy()
			tmp = np.repeat(tmp, 3, axis=1)
			segmentation_preds = tmp.astype('float32')
		# print("******************************************************")
		# print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
		# if output_depths != None:
		#     print(depth_truths.shape, depth_truths.mean().item(), depth_truths.max().item(), depth_truths.min().item())
		#     print(depth_preds.shape, depth_preds.mean().item(), depth_preds.max().item(), depth_preds.min().item())
		# if output_segmentations != None:
		#     print(segmentation_truths.shape, segmentation_truths.mean().item(), segmentation_truths.max().item(), segmentation_truths.min().item())
		#     print(segmentation_preds.shape, segmentation_preds.mean().item(), segmentation_preds.max().item(), segmentation_preds.min().item())
		# print("******************************************************")
		imgs = imgs.transpose(0, 2, 3, 1)
		if output_depths != None:
			depth_truths = depth_truths.transpose(0, 2, 3, 1)
			depth_preds = depth_preds.transpose(0, 2, 3, 1)
		if output_segmentations != None:
			segmentation_truths = segmentation_truths.transpose(0, 2, 3, 1)
			segmentation_preds = segmentation_preds.transpose(0, 2, 3, 1)
		output_dim = (
		int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))

		wandb.log({
			"img": [wandb.Image(cv2.resize(im, output_dim),
								caption='img_{}'.format(i + 1)) for i, im in
					enumerate(imgs)]
		})
		if output_depths != None:
			wandb.log({
				"depth_truths": [wandb.Image(cv2.resize(im, output_dim),
											 caption='depth_truths_{}'.format(
												 i + 1)) for i, im in
								 enumerate(depth_truths)],
				"depth_preds": [wandb.Image(cv2.resize(im, output_dim),
											caption='depth_preds_{}'.format(
												i + 1)) for i, im in
								enumerate(depth_preds)]
			})
		if output_segmentations != None:
			wandb.log({
				"seg_truths": [wandb.Image(cv2.resize(im, output_dim),
										   caption='seg_truths_{}'.format(
											   i + 1)) for i, im in
							   enumerate(segmentation_truths)],
				"seg_preds": [wandb.Image(cv2.resize(im, output_dim),
										  caption='seg_preds_{}'.format(i + 1))
							  for i, im in enumerate(segmentation_preds)]
			})
