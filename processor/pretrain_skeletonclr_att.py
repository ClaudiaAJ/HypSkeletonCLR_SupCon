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

from tools.losses import SupConLoss

import wandb

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 

from scipy.cluster.hierarchy import linkage, fcluster


def generate_pseudo_labels_from_attention(sp_attns, tp_attns, num_clusters):
        """
        Generate pseudo-labels using hierarchical clustering (Ward) on combined spatial
        and temporal attention maps.

        Args:
            sp_attns: list of spatial attention tensors, each of shape (N, heads, T, V, V)
            tp_attns: list of temporal attention tensors, each of shape (N, heads, V, T, T)
            num_clusters: int, number of clusters to form

        Returns:
            pseudo_labels: np.ndarray, shape (N,) with cluster assignments
        """

        # 1) Concatenate attention maps from all layers
        if isinstance(sp_attns[0], list):
            sp_attns = [item for sublist in sp_attns for item in sublist]
        if isinstance(tp_attns[0], list):
            tp_attns = [item for sublist in tp_attns for item in sublist]

        target_sp_shape = sp_attns[0].shape  # Now should be tensor with shape

        all_features = []
        for sp_att, tp_att in zip(sp_attns, tp_attns):
            # split batch into two views
            N = sp_att.shape[0]
            half = N // 2
            sp_pooled1 = sp_att[:half].mean(dim=(1,2,4))
            sp_pooled2 = sp_att[half:].mean(dim=(1,2,4))
            tp_pooled1 = tp_att[:half].mean(dim=(1,3,4))
            tp_pooled2 = tp_att[half:].mean(dim=(1,3,4))

            # average the two views
            sp_pooled = (sp_pooled1 + sp_pooled2) / 2
            tp_pooled = (tp_pooled1 + tp_pooled2) / 2

            combined = torch.cat([sp_pooled, tp_pooled], dim=1)
            all_features.append(combined)

        if len(all_features) == 0:
            raise ValueError("No attention layers with matching shape found!")

        # Stack features from all layers → (N, 500)
        features = torch.cat(all_features, dim=1).cpu().numpy()
        
        # 2) Hierarchical clustering with Ward linkage
        Z = linkage(features, method='ward')
        
        # 3) Cut dendrogram to form clusters
        cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
        
        return cluster_labels

class SkeletonCLR_Att_Processor(PT_Processor):
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

        self.criterion = SupConLoss(temperature=self.arg.temperature, curvature=self.arg.curvature)
        
    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        wandb.watch(self.model)
        # wandb.watch(self.model, log="all") # for logging of parameters panels

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
            if epoch < self.arg.sup_epoch:
                output, target, _, _, _ = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
            else:
                output, target, features_sup, sp_att, tp_att = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                # Generate pseudo-labels (example: 10 clusters)
                pseudo_labels = generate_pseudo_labels_from_attention(sp_att, tp_att, num_clusters=10)
                pseudo_labels = torch.tensor(pseudo_labels, device=self.dev, dtype=torch.long)
                
                alpha = 1.0
                loss = self.criterion(features_sup, pseudo_labels)

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

        if epoch < self.arg.sup_epoch:
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
        parser.add_argument('--curvature', type=float, default=1.0, help='the curvature of the Poincaré ball')
        
        # endregion yapf: enable

        return parser
