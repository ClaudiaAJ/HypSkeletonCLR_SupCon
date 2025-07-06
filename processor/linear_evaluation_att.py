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
import geoopt.manifolds.stereographic.math as pmath 

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import wandb

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LE_Processor(Processor):
    """
        Processor for Linear Evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize WandB
        wandb.init(
            project="HypSkeletonCLR_SupCon",
            config=vars(self.arg),  # Pass all arguments to wandb config
        )
        
        self.best_result = 0
        self.all_features = []
        self.all_labels = []

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

        for name, param in self.model.encoder_q.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        self.num_grad_layers = 2
        if hasattr(self.model, 'encoder_q_motion'):
            for name, param in self.model.encoder_q_motion.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            self.num_grad_layers += 2
        if hasattr(self.model, 'encoder_q_bone'):
            for name, param in self.model.encoder_q_bone.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            self.num_grad_layers += 2

        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(parameters) == self.num_grad_layers
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RSGD':
            self.optimizer = gt.optim.RiemannianSGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
        '''
        # Initialize CosineAnnealingLR after a warmup phase
        warmup_epochs = 0
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
        warmup_epochs = 0
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

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        self.model.eval()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        #poincare_ball = gt.PoincareBall(self.arg.curvature)

        for data, label in loader:
            self.global_step += 1
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            #data = poincare_ball.expmap0(data)

            # forward
            output, _, _ = self.model(data, view=self.arg.view)
            loss = self.loss(output, label)

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
                    "train_loss_le": loss.data.item(),
                    "learning_rate_le": self.lr,
                    "epoch_le": epoch},
                    step=self.global_step)
                
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)

        wandb.log({
                    "train_mean_loss_le": np.mean(loss_value),
                    "epoch_le": epoch},
                    step=self.global_step)
        
        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        #poincare_ball = gt.PoincareBall(self.arg.curvature)

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            #data = poincare_ball.expmap0(data)

            # inference
            with torch.no_grad():
                output, _, _ = self.model(data, view=self.arg.view)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())
        
        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss']= np.mean(loss_value)
        self.show_eval_info()

        # show top-k accuracy 
        for k in self.arg.show_topk:
            self.show_topk(k)
        self.show_best(1)

        wandb.log({
            "best_accuracy": self.best_result,
            "eval_mean_loss": np.mean(loss_value),
            "epoch_le": epoch},
            step=self.global_step)
        
        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')
        parser.add_argument('--curvature', type=float, default=1.0, help='the curvature of the Poincaré ball')
        
        # endregion yapf: enable

        return parser
