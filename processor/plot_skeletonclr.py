#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse

from hyperbolicTSNE import SequentialOptimizer, initialization, HyperbolicTSNE

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

import geoopt as gt


def visualize_latent_space(features, labels, method='pca', n_components=2, random_state=42, save_path=None, selected_labels=None):

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Filter features and labels for selected actions
    if selected_labels is not None:
        mask = np.isin(labels, selected_labels)
        features = features[mask]
        labels = labels[mask]
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
    elif method == 'hyp_tsne':
        #features_embedded = initialization(n_samples=features.shape[0], n_components=n_components,
        #                            X=features, random_state=random_state, method="pca")
        reducer = HyperbolicTSNE()
        #features = np.array(features, dtype=object) 

    #X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    #print("X shape:", X.shape)
    #features = X
    reduced_features = reducer.fit_transform(features)
    #print("reduced features shape:", reduced_features.shape)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", len(np.unique(labels)))
    sns.scatterplot(
        x=reduced_features[:, 0],
        y=reduced_features[:, 1],
        hue=labels,
        palette=palette,
        legend=True,
        alpha=0.7
    )
    if method == 'pca':
        plt.title(f"PCA of Model Output Features")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    
    elif method == 'svd':
        plt.title(f"SVD of Model Output Features")
        plt.xlabel("Singular Value 1")
        plt.ylabel("Singular Value 2")
    
    elif method == 'tsne':
        plt.title(f"t-SNE of Model Output Features")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

    elif method == 'hyp_tsne':
        plt.title(f"Hyperbolic t-SNE of Model Output Features")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

class SkeletonCLR_Plotting(PT_Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.poincare_ball = gt.PoincareBall(self.arg.curvature)

        self.all_features = []
        self.all_labels = []

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        features = []
        label_frag = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                latent_features = self.model.encoder_q(data)
                #latent_features = F.normalize(latent_features, dim=1)
                latent_features = self.poincare_ball.expmap0(latent_features)
                features.append(latent_features.cpu().numpy())
            
            label_frag.append(label.data.cpu().numpy())
        
        self.label = np.concatenate(label_frag)

        features = np.concatenate(features)

        self.all_features.append(features)
        self.all_labels.append(self.label)

        print("Generating plots with model output features...")
        save_path_svd = f"latent_space_svd.png"
        save_path_pca = f"latent_space_pca.png"
        save_path_tsne = f"latent_space_tsne.png"
        save_path_hyp_tsne = f"latent_space_hyperbolic_tsne.png"

        #selected_labels = [0, 5, 11, 17, 23, 29, 35, 41, 47, 53]
        selected_labels = [0, 5, 11, 17, 23, 26, 34, 35, 43, 54]
        visualize_latent_space(self.all_features, self.all_labels, method='svd', save_path=save_path_svd, selected_labels=selected_labels)
        visualize_latent_space(self.all_features, self.all_labels, method='pca', save_path=save_path_pca, selected_labels=selected_labels)
        visualize_latent_space(self.all_features, self.all_labels, method='tsne', save_path=save_path_tsne, selected_labels=selected_labels)
        visualize_latent_space(self.all_features, self.all_labels, method='hyp_tsne', save_path=save_path_hyp_tsne, selected_labels=selected_labels)
    
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
