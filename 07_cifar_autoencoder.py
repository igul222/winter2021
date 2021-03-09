"""
Semi-supervised classification from autoencoder representations on CIFAR-10.
"""

import argparse
import lib.ops
import lib.utils
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn, optim
from functools import partial

BATCH_SIZE = 128
AUGMENT_BS = 16
STEPS = 20*1000

parser = argparse.ArgumentParser()
parser.add_argument('--noise', type=float, default=0.3)
parser.add_argument('--rotation', type=float, default=0.)
parser.add_argument('--hue', type=float, default=0.)
parser.add_argument('--augment_cond', action='store_true')
parser.add_argument('--augment_mode', default='none')
args = parser.parse_args()
print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
    download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=8, drop_last=True)
def _():
    while True:
        yield from loader
inf_loader = _()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = lib.ops.WideResnet()
    def forward(self, x):
        z = self.encoder(x)
        z = z / z.pow(2).mean(dim=1, keepdim=True).sqrt()
        z_noisy = z + (args.noise * torch.randn_like(z))
        return z, z_noisy

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = lib.ops.WideResnetDecoder()
        self.theta_proj = nn.Linear(2, 256)
    def forward(self, z_noisy, theta):
        x = z_noisy
        if args.augment_cond:
            x = x + self.theta_proj(theta)
        x = self.decoder(x)
        return x

encoder = Encoder().cuda()
decoder = Decoder().cuda()

def _apply(x, scale, fn):
    """
    Shuffle x, apply fn on groups of consecutive samples, then unshuffle.
    """
    perm = torch.randperm(x.shape[0], device='cuda')
    inv_perm = torch.argsort(perm)
    x = x[perm]
    theta = torch.rand(1+(x.shape[0]//AUGMENT_BS), device='cuda')*2 - 1
    theta = theta.repeat_interleave(AUGMENT_BS)[:x.shape[0]]
    x = torch.cat([
        fn(x[i:i+AUGMENT_BS], (theta[i] * scale).item())
        for i in range(0, x.shape[0], AUGMENT_BS)
    ], dim=0)
    x, theta = x[inv_perm], theta[inv_perm]
    return x, theta

def augment(x):
    x, theta1 = _apply(x, args.rotation * 180., partial(T.functional.rotate,
        interpolation=T.InterpolationMode.BILINEAR))
    x, theta2 = _apply(x, args.hue * 0.5, T.functional.adjust_hue)

    theta = torch.stack([theta1, theta2], dim=1)
    return x, theta

def forward():
    x, y = next(inf_loader)
    x, y = x.cuda(), y.cuda()

    if args.augment_mode == 'none':
        x_enc = x
        x_dec = x
        theta = torch.zeros((x.shape[0], 1), device='cuda')
    elif args.augment_mode == 'encoder_only':
        x_dec = x
        x_enc, _ = augment(x)
        theta = torch.zeros((x.shape[0], 1), device='cuda')
    elif args.augment_mode == 'decoder_only':
        x_enc = x
        x_dec, theta = augment(x)
    elif args.augment_mode == 'same':
        x_enc, theta = augment(x)
        x_dec = x_enc
    elif args.augment_mode == 'different':
        x_enc, _ = augment(x)
        x_dec, theta = augment(x)

    # Rescale to [-1, 1]
    x_enc = (2*x_enc) - 1
    x_dec = (2*x_dec) - 1

    _, z_noisy = encoder(x_enc)
    x_reconst = decoder(z_noisy, theta.cuda())
    loss = (x_dec - x_reconst).pow(2).mean()
    return loss
opt = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=3e-4)

def extract_feats(train):
    dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
        transform=torchvision.transforms.ToTensor(), train=train)
    loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        Z, Y = [], []
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            x = (2*x) - 1 # Rescale to [-1, 1]
            z, _ = encoder(x)
            Z.append(z)
            Y.append(y)
        Z = torch.cat(Z, dim=0)
        Y = torch.cat(Y, dim=0)
    return Z, Y

def run_eval():
    # Step 1: Train a linear classifier
    Z, Y = extract_feats(train=True)
    linear_model = nn.Linear(Z.shape[1], 10).cuda()
    def forward():
        return F.cross_entropy(linear_model(Z), Y)
    opt = optim.Adam(linear_model.parameters(), lr=1e-3)
    lib.utils.train_loop(forward, opt, 10*1000, quiet=True)
    # Step 2: Evaluate
    Z, Y = extract_feats(train=False)
    with torch.no_grad():
        y_pred = linear_model(Z).argmax(dim=1)
        acc = y_pred.eq(Y).float().mean()
    print('Test acc:', acc.item())

lib.utils.train_loop(forward, opt, STEPS)
run_eval()