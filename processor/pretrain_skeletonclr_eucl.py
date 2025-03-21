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

from tools.losses_eucl import SupConLoss

import wandb

class SkeletonCLR_Processor(PT_Processor):
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
        })

        self.criterion = SupConLoss(temperature=self.arg.temperature)

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        wandb.watch(self.model)
        # wandb.watch(self.model, log="all") # for logging of parameters panels

        label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1,
                        10: 2, 11: 3, 12: 0, 13: 1, 14: 2, 15: 3, 16: 0, 17: 1, 18: 2, 19: 3,
                        20: 0, 21: 1, 22: 2, 23: 3, 24: 0, 25: 1, 26: 2, 27: 3, 28: 0, 29: 1,
                        30: 2, 31: 3, 32: 0, 33: 1, 34: 2, 35: 3, 36: 0, 37: 1, 38: 2, 39: 3,
                        40: 0, 41: 1, 42: 2, 43: 3, 44: 0, 45: 1, 46: 2, 47: 3, 48: 0, 49: 1,
                        50: 2, 51: 3, 52: 0, 53: 1, 54: 2, 55: 3, 56: 0, 57: 1, 58: 2, 59: 3}

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.view == 'joint':
                pass
            elif self.arg.view == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
            elif self.arg.view == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
                
                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                
                data1 = bone1
                data2 = bone2
            else:
                raise ValueError

            # forward
            #if epoch <= self.arg.sup_epoch:
            if epoch < self.arg.sup_epoch:
                output, target, _ = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
            else:
                output, target, features_sup = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                
                loss_unsup = self.loss(output, target)
                
                try:
                    label_sup = torch.tensor([label_mapping[int(l)] for l in label])
                except NameError:
                    label_sup = label
                    
                loss_sup = self.criterion(features_sup, label_sup)
                
                # new loss function: scaled sum of unsupervised and supervised loss
                #alpha = (epoch - self.arg.sup_epoch) / (self.arg.num_epoch - self.arg.sup_epoch)
                alpha = 1.0
                loss = (1 - alpha) * loss_unsup + alpha * loss_sup

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

            if self.global_step % 100 == 0:
                # Log metrics to wandb
                wandb.log({
                    "loss": loss.data.item(),
                    #"supervised_loss": loss_sup.data.item(),
                    #"unsupervised_loss": loss_unsup.data.item(),
                    "learning_rate": self.lr,
                    "epoch": epoch},
                    step=self.global_step)
            
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)

        if epoch <= self.arg.sup_epoch:
            alpha = 0
            print(f"Scaling of Loss Functions -> Unsupervised: {1 - alpha:.4f}, Supervised: {alpha:.4f}")
        else:
            print(f"Scaling of Loss Functions -> Unsupervised: {1 - alpha:.4f}, Supervised: {alpha:.4f}")
        
        # Log epoch-level mean loss
        wandb.log({
            "train_mean_loss": np.mean(loss_value),
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
        
        # endregion yapf: enable

        return parser
