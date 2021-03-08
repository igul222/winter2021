"""
Semi-supervised classification from autoencoder representations on CIFAR-10.
"""

import argparse
import lib.ops
import lib.utils
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn, optim

BATCH_SIZE = 128
AUGMENT_BS = 16
STEPS = 20*1000

parser = argparse.ArgumentParser()
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--max_rotation', type=float, default=0.)
parser.add_argument('--augment_cond', action='store_true')
args = parser.parse_args()
print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
    download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=4, drop_last=True)
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
        self.theta_proj = nn.Linear(1, 256)
    def forward(self, z_noisy, theta):
        x = z_noisy
        if args.augment_cond:
            x = x + self.theta_proj(theta)
        x = self.decoder(x)
        return x

encoder = Encoder().cuda()
decoder = Decoder().cuda()

def forward():
    x, y = next(inf_loader)
    x, y = x.cuda(), y.cuda()
    x = (2*x) - 1 # Rescale to [-1, 1]

    theta = (torch.rand(AUGMENT_BS, 1)*2-1)
    theta = theta.repeat(BATCH_SIZE//AUGMENT_BS, 1)
    x = torch.cat([
        T.functional.rotate(
            x[i::AUGMENT_BS],
            (theta[i] * args.max_rotation).item(),
            interpolation=T.InterpolationMode.BILINEAR
        ) for i in range(AUGMENT_BS)], dim=0)

    _, z_noisy = encoder(x)
    x_reconst = decoder(z_noisy, theta.cuda())
    loss = (x - x_reconst).pow(2).mean()
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