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
STEPS = 40001
PRINT_FREQ = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--augment_cond', action='store_true')
parser.add_argument('--augment_mode', default='none')
parser.add_argument('--resnet_n', type=int, default=1)
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--noise', type=float, default=0.3)
parser.add_argument('--rotation', type=float, default=0.)
parser.add_argument('--hue', type=float, default=0.)
parser.add_argument('--q_levels', type=int, default=2048)
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
        self.encoder = lib.ops.WideResnet(N=args.resnet_n, k=args.k)
    def forward(self, x):
        z = self.encoder(x)
        z = z / z.pow(2).mean(dim=1, keepdim=True).sqrt()
        z_noisy = z + (args.noise * torch.randn_like(z))
        return z, z_noisy

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = lib.ops.WideResnetDecoder(N=args.resnet_n, k=args.k, dim_out=16*args.k)
        self.theta_proj = nn.Linear(2, 64*args.k)
        self.norm = nn.GroupNorm(8, 16*args.k)
        self.embedding = nn.Embedding(args.q_levels, 16*args.k)

        LSTM_DIM = 512
        self.lstm = nn.LSTM(
            input_size=16*args.k,
            hidden_size=LSTM_DIM,
            num_layers=1,
            batch_first=True
        )
        self.lstm = nn.utils.weight_norm(self.lstm, 'weight_ih_l0')
        self.lstm = nn.utils.weight_norm(self.lstm, 'weight_hh_l0')
        self.readout = nn.Linear(LSTM_DIM, args.q_levels)

    def forward(self, z_noisy, theta, x_target):
        x = z_noisy
        if args.augment_cond:
            x = x + self.theta_proj(theta)
        x = self.decoder(x)

        x = self.norm(x)

        x = x.view(x.shape[0],16*args.k,32*32).permute(0,2,1)
        x_target_embed = self.embedding(x_target)
        x_target_embed = torch.cat([
            torch.zeros_like(x_target_embed[:,0:1,:]),
            x_target_embed[:,:-1,:]
        ], dim=1)
        x = x + x_target_embed
        x, _ = self.lstm(x)
        x = self.readout(x)

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

def quantize(x):
    """
    input: (n, 3, 32, 32) floats in [0, 1]
    output: (n, 32*32) ints in [0, Q_LEVELS)
    """
    q_levels = int(np.cbrt(args.q_levels))
    x = x*255/256 # [0, 1)
    x = (x * q_levels) # [0, q_levels)
    x = x.long() # ints in [0, q_levels) (*biased by 0.5 downwards)
    x = (q_levels**2 * x[:,0,:,:]) + (q_levels * x[:,0,:,:]) + x[:,2,:,:]
    x = x.view(x.shape[0], 32*32)
    return x

def dequantize(x):
    """
    input: (n, 32*32) ints in [0, Q_LEVELS)
    output: (n, 3, 32, 32) floats in [0, 1]
    """
    q_levels = int(np.cbrt(args.q_levels))
    x = x.view(x.shape[0], 32, 32)
    x0 = (x // q_levels**2) % q_levels
    x1 = (x // q_levels) % q_levels
    x2 = x % q_levels
    x = torch.stack([x0, x1, x2], dim=1) # (n, 3, 32, 32) in [0, q_levels)    
    x = (x.float() + 0.5) / q_levels # bias-corrected and scaled to [0, 1)
    return x

def forward():
    x, y = next(inf_loader)
    x, y = x.cuda(), y.cuda()

    if args.augment_mode == 'none':
        x_enc = x
        x_dec = x
        theta = torch.zeros((x.shape[0], 2), device='cuda')
    elif args.augment_mode == 'encoder_only':
        x_dec = x
        x_enc, _ = augment(x)
        theta = torch.zeros((x.shape[0], 2), device='cuda')
    elif args.augment_mode == 'decoder_only':
        x_enc = x
        x_dec, theta = augment(x)
    elif args.augment_mode == 'same':
        x_enc, theta = augment(x)
        x_dec = x_enc
    elif args.augment_mode == 'different':
        x_enc, _ = augment(x)
        x_dec, theta = augment(x)

    x_enc = (2*x_enc) - 1
    x_dec = quantize(x_dec)

    _, z_noisy = encoder(x_enc)
    x_reconst = decoder(z_noisy, theta.cuda(), x_dec)
    loss = F.cross_entropy(x_reconst.view(-1, args.q_levels),
        x_dec.view(-1))
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

def run_eval(step):
    if step % 20000 != 0:
        return
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

lib.utils.train_loop(forward, opt, STEPS, hook=run_eval, print_freq=PRINT_FREQ)