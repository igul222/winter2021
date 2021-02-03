"""
My version of a TDRE toy problem: separated Gaussians, piecewise-uniform
density ratio function.
"""

import lib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn, optim

N = 10*1000
PQ_DISTANCE = 10
N_BINS = 100

p = torch.distributions.normal.Normal(-PQ_DISTANCE/2, 1.)
q = torch.distributions.normal.Normal(PQ_DISTANCE/2, 1.)

x_p = p.sample(sample_shape=[N])
x_q = q.sample(sample_shape=[N])

class PiecewiseUniform(nn.Module):
    def __init__(self, min_, max_, n_bins):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros((n_bins,)))
        self.n_bins = n_bins
        self.min_ = min_
        self.max_ = max_
    def forward(self, x):
        x = x - self.min_
        x = x / (self.max_ - self.min_)
        x = x * self.n_bins
        x = torch.clamp(x, min=0, max=self.n_bins-1).long()
        x = self.theta[x.flatten()].view(*x.shape)
        return x

def save_plot(name, dists, models):
    plt.clf()
    for dist in dists:
        plt.hist(dist, bins=20, alpha=0.1, density=True)
    x = torch.arange(-PQ_DISTANCE/2 - 2, PQ_DISTANCE/2 + 2, 0.05)
    for model in models:
        y = torch.sigmoid(model(x))
        plt.scatter(x.detach(), y.detach())
    plt.savefig(name+'.png')

def train_model(x_p, x_q):
    model = PiecewiseUniform(-PQ_DISTANCE/2 - 2, PQ_DISTANCE/2 + 2, N_BINS)
    def forward():
        bce = F.binary_cross_entropy_with_logits
        return (
            bce(model(x_p), torch.ones_like(x_p))
            + bce(model(x_q), torch.zeros_like(x_q))
        ) / 2.
    opt = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    lib.utils.train_loop(forward, opt, 3001)
    return model

# Plot 1: model trained with binary cross-entropy

model = train_model(x_p, x_q)
save_plot('bce', [x_p, x_q], [model])

# Plot 2: model trained with TDRE

def interpolate(alpha, x_p, x_q):
    return x_p + alpha*(x_q - x_p)

alphas = torch.arange(0., 1. + 1e-6, 0.05)
waypoints = [interpolate(alpha, x_p, x_q) for alpha in alphas]
models = [train_model(x1, x2) for x1, x2 in zip(waypoints[:-1], waypoints[1:])]
def tdre_model(x):
    return torch.stack([m(x) for m in models], dim=0).sum(dim=0)

save_plot('tdre', waypoints, [tdre_model])

# Plot 3: logit at 0, across all the waypoint-models

logits = [m(torch.tensor([0.]))[0].item() for m in models]
plt.clf()
plt.scatter(alphas[:-1].numpy(), np.array(logits))
plt.xlabel('alpha')
plt.ylabel('logit(0)')
plt.savefig('logit_vs_alpha.png')