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

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize, StandardScaler

def visualize_latent_space(features, labels, method='pca', n_components=2, random_state=42, save_path=None, selected_labels=None):
    
    print(f"Normalizing features for {method}...")
    features = normalize(features, axis=1)
    print(f"Features normalized.")
    
    # Filter features and labels for selected actions
    if selected_labels is not None:
        mask = np.isin(labels, selected_labels)
        features = features[mask]
        labels = labels[mask]

    #'''
    label_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1,
                    10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 16: 0, 17: 1, 18: 0, 19: 1,
                    20: 0, 21: 1, 22: 0, 23: 1, 24: 0, 25: 1, 26: 0, 27: 1, 28: 0, 29: 1,
                    30: 0, 31: 1, 32: 0, 33: 1, 34: 0, 35: 1, 36: 0, 37: 1, 38: 0, 39: 1,
                    40: 0, 41: 1, 42: 0, 43: 1, 44: 0, 45: 1, 46: 0, 47: 1, 48: 0, 49: 1,
                    50: 0, 51: 1, 52: 0, 53: 1, 54: 0, 55: 1, 56: 0, 57: 1, 58: 0, 59: 1}
    labels = torch.tensor([label_mapping[int(l)] for l in labels])
    #'''
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Unsupported method. Choose either 'pca', 'svd' or 'tsne'.")

    reduced_features = reducer.fit_transform(features)

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

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}.")
    
    plt.show()

class SkeletonCLR_Plotting(PT_Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.all_features = []
        self.all_labels = []

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        features = []
        labels = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                latent_features = self.model.encoder_q(data)
                latent_features = F.normalize(latent_features, dim=1)
                features.append(latent_features.cpu().numpy())
            
            labels.append(label.data.cpu().numpy())

        self.all_labels = np.concatenate(labels)
        self.all_features = np.concatenate(features)

        print("Generating plots with model output features...")
        save_path_svd = f"latent_space_svd_eucl.png"
        save_path_pca = f"latent_space_pca_eucl.png"
        save_path_tsne = f"latent_space_tsne_eucl.png"

        #selected_labels = [0, 5, 11, 17, 23, 26, 34, 35, 43, 54]
        #selected_labels = [5, 6, 9, 13, 14, 25, 39, 42, 50, 54]
        #selected_labels = [5, 13, 14, 25, 27, 39, 42, 50, 52, 54]
        selected_labels = [5, 11, 13, 14, 25, 27, 39, 42, 50, 54]

        #visualize_latent_space(self.all_features, self.all_labels, method='svd', save_path=save_path_svd, selected_labels=selected_labels)
        #visualize_latent_space(self.all_features, self.all_labels, method='pca', save_path=save_path_pca, selected_labels=selected_labels)
        visualize_latent_space(self.all_features, self.all_labels, method='tsne', save_path=save_path_tsne, selected_labels=selected_labels)
    
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
        parser.add_argument('--temperature', type=float, default=0.07, help='the temperature used in supervised training loss')
        
        # endregion yapf: enable

        return parser
