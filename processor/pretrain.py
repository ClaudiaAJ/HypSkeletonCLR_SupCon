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
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

import geoopt as gt

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class PT_Processor(Processor):
    """
        Processor for Pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RSGD':
            self.optimizer = gt.optim.RiemannianSGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        '''
        # Initialize CosineAnnealingLR after a warmup phase
        warmup_epochs = 15
        num_cycles = 1
        
        # Compute the duration of each cycle
        remaining_epochs = self.arg.num_epoch - warmup_epochs
        cycle_epochs = remaining_epochs // num_cycles
        if self.arg.num_epoch > 0: 
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return epoch / warmup_epochs
                else:
                    # Determine the current cycle
                    adjusted_epoch = epoch - warmup_epochs
                    current_cycle = adjusted_epoch // cycle_epochs
                    cycle_position = adjusted_epoch % cycle_epochs
                    if current_cycle >= num_cycles:
                        return 0  # Learning rate reaches zero after the last cycle
                    
                    # Compute the annealing factor for the current cycle
                    cycle_start_lr = self.arg.base_lr * (0.5 ** current_cycle)  # Halve base_lr each cycle
                    min_lr = cycle_start_lr * 0.1  # Set the minimum lr to 10% of the cycle's start_lr
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_position / cycle_epochs))
                    return (min_lr + (cycle_start_lr - min_lr) * cosine_decay) / self.arg.base_lr
            
            # Create the LambdaLR scheduler with the custom function
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.lr_scheduler = None  # No scheduler if num_epoch is not defined
        '''
        # Initialize CosineAnnealingLR after a warmup phase
        warmup_epochs = 15
        if self.arg.num_epoch > 0: 
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return epoch / warmup_epochs
                else:
                    # Scale for cosine annealing (after warmup)
                    cosine_scheduler = CosineAnnealingLR(self.optimizer,T_max=self.arg.num_epoch - warmup_epochs,eta_min=0)
                    cosine_scheduler.step(epoch - warmup_epochs)  # Adjust for post-warmup epochs
                    return self.optimizer.param_groups[0]['lr'] / self.arg.base_lr
            # Combine warmup and cosine annealing with LambdaLR
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.lr_scheduler = None  # No scheduler if num_epoch is not defined
        
    def adjust_lr_scheduler(self):
        # Using CosineAnnealingLR scheduler with warmup
        if self.lr_scheduler:
            self.lr_scheduler.step()  # Update the learning rate based on the current epoch
            self.lr = self.lr_scheduler.get_last_lr()[0]  # Get the updated learning rate
        else:
            self.lr = self.arg.base_lr

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            output, target = self.model(data1, data2)
            loss = self.loss(output, target)

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
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
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
        # endregion yapf: enable

        return parser
