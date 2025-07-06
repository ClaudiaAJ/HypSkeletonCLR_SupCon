#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse

#from hyperbolicTSNE import SequentialOptimizer, initialization, HyperbolicTSNE
#from hyperbolicTSNE import hd_mat_ as hd_mat
from scipy.sparse import csr_matrix

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

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 


class SkeletonCLR_Plotting(PT_Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, epoch):
        self.model.eval()
        loader = self.data_loader['train']
        features_list = []
        labels_list = []

        with torch.no_grad():
            for batch in loader:
                (x, _), labels = batch
                data = x
                data = data.float().cuda()

                feats = self.model.encoder_q(data)          
                feats = feats.cpu().numpy()

                features_list.append(feats)
                if labels is not None:
                    labels_list.append(labels.numpy())

        features = np.concatenate(features_list, axis=0)
        print("features shape:", features.shape)
        
        if labels_list:
            labels_all = np.concatenate(labels_list, axis=0)
            print("labels shape:", labels_all.shape)

        features_tensor = torch.from_numpy(features).float()
        features_norm = F.normalize(features_tensor, p=2, dim=1)

        poincare_ball = gt.PoincareBall(self.arg.curvature)
        features_norm = poincare_ball.expmap0(features_norm)
        features_norm_np = features_norm.numpy()

        subset_size = 10000
        if features_norm_np.shape[0] > subset_size:
            rng = np.random.RandomState(0)
            idx = rng.choice(features_norm_np.shape[0], subset_size, replace=False)
            feats = features_norm_np[idx]
            labs = labels_all[idx]
        else:
            feats = features_norm_np
            labs = labels_all
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        emb = tsne.fit_transform(feats)

        print("Generating plots with model output features...")
        save_path_tsne = f"latent_space_tsne_mert.png"

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labs, cmap='tab20', s=5)
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.title('t-SNE of L2-normalized Features Colored by Label')
        plt.colorbar(scatter, label='Label')
        plt.tight_layout()
        plt.savefig(save_path_tsne, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {save_path_tsne}.")
        
    
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
