#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor

import wandb

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 

class SkeletonCLR_3views_Processor(PT_Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize wandb run
        wandb.init(project="HypSkeletonCLR_SupCon")
        wandb.config.update({
            "learning_rate": self.arg.base_lr,
            "optimizer": self.arg.optimizer,
            "weight_decay": self.arg.weight_decay,
            "nesterov": self.arg.nesterov,
            "num_epochs": self.arg.num_epoch,
            "sup_epoch": self.arg.sup_epoch,
            "temperature": self.arg.temperature,
            "curvature": self.arg.curvature,
        })

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_motion_value = []
        loss_bone_value = []

        poincare_ball = gt.PoincareBall(self.arg.curvature)

        wandb.watch(self.model)
        # wandb.watch(self.model, log="all") # for logging of parameters panels

        for [data1, data2], label in loader:
            self.global_step += 1

            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            output, output_motion, output_bone, target = self.model(data1, data2)
            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(output.size(0))
            else:
                self.model.update_ptr(output.size(0))
            loss = self.loss(output, target)
            loss_motion = self.loss(output_motion, target)
            loss_bone = self.loss(output_bone, target)

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_motion'] = loss_motion.data.item()
            self.iter_info['loss_bone'] = loss_bone.data.item()
            loss_value.append(self.iter_info['loss'])
            loss_motion_value.append(self.iter_info['loss_motion'])
            loss_bone_value.append(self.iter_info['loss_bone'])
            loss = loss + loss_motion + loss_bone

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_writer.add_scalar('batch_loss_motion', self.iter_info['loss_motion'], self.global_step)
            self.train_writer.add_scalar('batch_loss_bone', self.iter_info['loss_bone'], self.global_step)

            if self.global_step % 100 == 0:
                # Log metrics to wandb
                wandb.log({
                    "loss": loss.data.item(),
                    "loss_motion": loss_motion.data.item(),
                    "loss_bone": loss_bone.data.item(),
                    "learning_rate": self.lr,
                    "epoch": epoch},
                    step=self.global_step)
            
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)

        # Log epoch-level mean loss
        wandb.log({
            "train_mean_loss": np.mean(loss_value),
            "train_mean_loss_motion": np.mean(loss_motion_value),
            "train_mean_loss_bone": np.mean(loss_bone_value),
            "learning_rate": self.lr,
            "epoch": epoch},
            step=self.global_step)

        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--sup_epoch', type=int, default=1e6, help='the starting epoch of supervised training')
        parser.add_argument('--temperature', type=float, default=0.07, help='the temperature used in supervised training loss')
        parser.add_argument('--curvature', type=float, default=1.0, help='the curvature of the Poincaré ball')
        
        # endregion yapf: enable

        return parser
