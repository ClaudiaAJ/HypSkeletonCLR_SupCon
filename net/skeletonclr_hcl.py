import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
# HYP: libraries
import geoopt as gt
#import geoopt.manifolds.stereographic.math as pmath 

import tools.hyptorch.pmath as pmath

import tools.hyptorch.nn as hypnn

class SkeletonCLR_HCL(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, curvature=1.0, train_x=False, train_c=False, decoupled=False, **kwargs):
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
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.c = curvature
            self.decoupled = decoupled

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
            
            dim_fc_out = self.encoder_q.fc.weight.shape[0]
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
            
            #self.tp = hypnn.ToPoincare(c=curvature, train_x=train_x, train_c=train_c, ball_dim=dim_fc_out, riemannian=False) # no RSGD
            self.tp = hypnn.ToPoincare(c=curvature, train_x=train_x, train_c=train_c, ball_dim=dim_fc_out)
            self.hyperbolic_dis = hypnn.HyperbolicDistanceLayer(c=curvature)

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, self.K))
            self.queue = nn.functional.normalize(self.queue, dim=0) 
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
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    def decouple_norm_from_direction(self, v):
        v_hat = nn.functional.normalize(v, dim=1).clone().detach()
        v_norm = v.norm(2, dim=1, keepdim=True)
        return v_hat*v_norm
    
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

        if cross:
            return self.cross_training(im_q, im_k, topk, context)

        if not self.pretrain:
            return self.encoder_q(im_q)

        #q_hyp_p = self.encoder_q(im_q)  # queries: NxC
        q_hyp_c = self.encoder_q(im_q)  # queries: NxC

        if self.decoupled:
            q_hyp_c_Rn = self.decouple_norm_from_direction(q_hyp_c)
        else:
            q_hyp_c_Rn = q_hyp_c
        qc_proj = self.tp(q_hyp_c_Rn) # Nx1

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k_hyp_p = self.encoder_k(im_k)  # keys: NxC

            kp_proj= self.tp(k_hyp_p)

        l_pos_h = -self.hyperbolic_dis(qc_proj, kp_proj)
        l_neg_h = -self.hyperbolic_dis(qc_proj.unsqueeze(1), self.queue.clone().detach().T.unsqueeze(0)).squeeze()
        logits_h = torch.cat([l_pos_h, l_neg_h], dim=1)
        logits_h /= self.T
        labels_h = torch.zeros(logits_h.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(kp_proj)
        
        features = torch.cat([qc_proj.unsqueeze(1), kp_proj.unsqueeze(1)], dim=1)

        return logits_h, labels_h, features
        