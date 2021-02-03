"""
Reproducing the 1D toy problem from the Telescoping Density Ratio Estimation
paper.
"""

import numpy as np
import lib
import torch
import torch.nn.functional as F
from torch import nn, optim

def log_ratio(theta, x, std_ratio):
    """
    Log-ratio of p(x)/q(x). This seems to match Fig 1 in the paper. Where the
    factor of 2 in theta came from, I don't know.
    """
    return torch.log(torch.tensor(std_ratio)) - torch.exp(2*theta) * x**2

def logistic_loss(theta, x_p, x_q, std_ratio):
    log_r_p = log_ratio(theta, x_p, std_ratio)
    log_r_q = log_ratio(theta, x_q, std_ratio)
    loss = (
        F.binary_cross_entropy_with_logits(log_r_p, torch.ones_like(log_r_p))
        + F.binary_cross_entropy_with_logits(log_r_q, torch.zeros_like(log_r_q))
    )
    return loss

def argmin_logistic_loss(x_p, x_q, std_ratio):
    min_loss, min_theta = np.inf, None
    for theta in torch.arange(4, 15, 0.1):
        loss = logistic_loss(theta, x_p, x_q, std_ratio)
        if loss < min_loss:
            min_loss, min_theta = loss, theta
    return min_theta

print('BCE, theta vs. N:')
lib.utils.print_row('N', 'theta')
for N in [10, 100, 1000, 10*1000, 100*1000]:
    x_p = 1e-6 * torch.randn((N,))
    x_q = torch.randn((N,))
    lib.utils.print_row(N, argmin_logistic_loss(x_p, x_q, 1e6))

print('BCE, theta std at N=10K:')
N = 10*1000
thetas = []
for _ in range(10):
    x_p = 1e-6 * torch.randn((N,))
    x_q = torch.randn((N,))
    theta = argmin_logistic_loss(x_p, x_q, 1e6)
    thetas.append(theta)
lib.utils.print_row('mean', 'std')
lib.utils.print_row(np.mean(thetas), np.std(thetas))

def make_waymark(x_p, x_q, alpha, std_p, std_q):
    x_waymark = (1-alpha)*x_p + alpha*x_q
    std_waymark = np.sqrt(((1-alpha)**2)*(std_p**2) + (alpha**2)*(std_q**2))
    return x_waymark, std_waymark

N = 1*1000
std_p = 1e-6
std_q = 1
x_p = std_p * torch.randn((N,))
x_q = std_q * torch.randn((N,))
x_1, std_1 = make_waymark(x_p, x_q, 0.0001, std_p, std_q)
x_2, std_2 = make_waymark(x_p, x_q, 0.001, std_p, std_q)
x_3, std_3 = make_waymark(x_p, x_q, 0.01, std_p, std_q)
x_4, std_4 = make_waymark(x_p, x_q, 0.1, std_p, std_q)
theta_1 = argmin_logistic_loss(x_p, x_1, std_1 / std_p)
theta_2 = argmin_logistic_loss(x_1, x_2, std_2 / std_1)
theta_3 = argmin_logistic_loss(x_2, x_3, std_3 / std_2)
theta_4 = argmin_logistic_loss(x_3, x_4, std_4 / std_3)
theta_5 = argmin_logistic_loss(x_4, x_q, std_q / std_4)
theta_tre = torch.logsumexp(torch.stack([theta_1, theta_2, theta_3, theta_4,
    theta_5]), 0)
print(f'TRE theta (N={N}): {theta_tre.item()}')