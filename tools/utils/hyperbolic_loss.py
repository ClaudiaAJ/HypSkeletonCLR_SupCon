import geoopt
import torch
import torch.nn.functional as F

# InfoNCE for first 150 epochs
def hyperbolic_infoNCE_loss(u, v, negatives, temperature=0.1):
    # u, v are positive pairs, negatives is a tensor of negative pairs
    # Embed in the Poincaré ball
    u_hyp = geoopt.manifolds.poincare.math.project(u)
    v_hyp = geoopt.manifolds.poincare.math.project(v)
    negatives_hyp = geoopt.manifolds.poincare.math.project(negatives)

    # Compute hyperbolic distances
    pos_distance = geoopt.manifolds.poincare.math.dist(u_hyp, v_hyp)
    neg_distances = geoopt.manifolds.poincare.math.dist(u_hyp.unsqueeze(1), negatives_hyp)

    # Apply the contrastive loss formula
    pos_term = torch.exp(pos_distance / temperature)
    neg_term = torch.exp(neg_distances / temperature).sum(dim=1)
    loss = -torch.log(pos_term / (pos_term + neg_term))

    return loss.mean()

# cross view loss for after 150th epoch 
def hyperbolic_cross_view_loss(q, k, queue, temperature):
    # Embed in the Poincaré ball
    q_hyp = geoopt.manifolds.poincare.math.project(q)
    k_hyp = geoopt.manifolds.poincare.math.project(k)
    queue_hyp = geoopt.manifolds.poincare.math.project(queue)

    # Compute hyperbolic distances
    pos_distance = geoopt.manifolds.poincare.math.dist(q_hyp, k_hyp)
    neg_distances = geoopt.manifolds.poincare.math.dist(q_hyp.unsqueeze(1), queue_hyp)

    # Apply the contrastive loss formula
    pos_term = torch.exp(pos_distance / temperature)
    neg_term = torch.exp(neg_distances / temperature).sum(dim=1)
    loss = -torch.log(pos_term / (pos_term + neg_term)) # TO-DO: update log term!

    return loss.mean()

def hyperbolic_cross_consistency_loss(q_views, k_views, queue_views, temperature):
    total_loss = 0.0
    U = len(q_views)
    for u in range(U):
        for v in range(U):
            if u != v:
                total_loss += hyperbolic_cross_view_loss(q_views[u], k_views[v], queue_views[v], temperature)
    return total_loss

