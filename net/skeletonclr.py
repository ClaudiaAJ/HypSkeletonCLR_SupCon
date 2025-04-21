import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
# HYP: libraries
import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 

#import tools.hyptorch.pmath as pmath

class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, curvature=1.0, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.c = curvature
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.c = curvature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, view='joint', cross=False, topk=1, context=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        # HYP: Initialize the Poincaré ball manifold
        poincare_ball = gt.PoincareBall(self.c)

        if cross:
            return self.cross_training(im_q, im_k, topk, context)

        if not self.pretrain:
            """
            Linear evaluation:
                perform same projections as in pretraining
            """
            #im_q = poincare_ball.expmap0(im_q)
            
            #im_q = poincare_ball.logmap0(im_q)

            #q = self.encoder_q(im_q)
            #q = poincare_ball.expmap0(q)

            #q = F.normalize(q, dim=1)
            #return q
            return self.encoder_q(im_q)
        
        """
        Pretraining:
            project features to poincare ball in hyperbolic space
        """
        #im_q = poincare_ball.expmap0(im_q)
        #im_q = poincare_ball.logmap0(im_q)

        # compute query features
        q = self.encoder_q(im_q)  # queries shape: [batch_size, feature_dim]
        q = F.normalize(q, dim=1)
        q = poincare_ball.expmap0(q) # shape: [batch_size, feature_dim]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            #im_k = poincare_ball.expmap0(im_k)
            #im_k = poincare_ball.logmap0(im_k)

            # compute key features
            k = self.encoder_k(im_k)  # keys shape: [batch_size, feature_dim]
            k = F.normalize(k, dim=1)
            k_eucl = k.clone().detach()
            k = poincare_ball.expmap0(k) # shape: [batch_size, feature_dim]
        
        # compute logits
        # positive logits shape: [batch_size, 1]
        l_pos = -poincare_ball.dist(q, k).unsqueeze(-1) 

        # negative logits shape: [batch_size, queue_size]
        # transpose self.queue to match dimensions for pairwise comparison [feature_dim, queue_size]
        # expand q and queue to compute pairwise distances
        # compute all pairwise (negative) hyperbolic distances between q and queue
        l_neg = -poincare_ball.dist(q.unsqueeze(1), poincare_ball.expmap0(self.queue.clone().detach().T))

        # logits shape: [batch_size, 1+queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Combine q (query) and k (key) as two views of the same image
        features = torch.cat([q.unsqueeze(1), k.unsqueeze(1)], dim=1)  # features shape: [batch_size, n_views, feature_dim], with n_views=2 (q and k)

        #queue = poincare_ball.expmap0(self.queue.clone().detach().T).unsqueeze(0) # sh [1, queue_size, feature_dim]
        #queue_reshaped = queue.expand(q.shape[0], -1, -1) # [batch_size, queue_size, feature_dim]
        #features = torch.cat([q.unsqueeze(1), queue_reshaped], dim=1) # features shape: [batch_size, n_views, feature_dim], with n_views=1+queue_size (q and queue)
        
        # dequeue and enqueue
        #self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue(k_eucl)

        return logits, labels, features
        