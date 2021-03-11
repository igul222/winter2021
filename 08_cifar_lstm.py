"""
Pure LSTM generative model on CIFAR-10, just to see how it does.
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
STEPS = 100*1000

dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
    download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=8, drop_last=True)
def _():
    while True:
        yield from loader
inf_loader = _()

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=1024,
            num_layers=1,
            batch_first=True
        )
        self.lstm = nn.utils.weight_norm(self.lstm, 'weight_ih_l0')
        self.lstm = nn.utils.weight_norm(self.lstm, 'weight_hh_l0')
        self.readout = nn.Linear(1024, 256)

    def forward(self, x_target):
        x_target_embed = self.embedding(x_target).permute(0,2,3,1,4).reshape(
            x_target.shape[0], 32*32*3, 64)
        x_target_embed = torch.cat([
            torch.zeros_like(x_target_embed[:,0:1,:]),
            x_target_embed[:,:-1,:]
        ], dim=1)
        x,_ = self.lstm(x_target_embed)
        x = self.readout(x)
        x = x.reshape(-1,32,32,3,256).permute(0,4,3,1,2)
        return x

decoder = Decoder().cuda()

def quantize(x):
    return (x*255.).long().clamp(min=0, max=255)

def forward():
    x, y = next(inf_loader)
    x, y = x.cuda(), y.cuda()
    x_quant = quantize(x)
    x_reconst = decoder(x_quant)
    loss = F.cross_entropy(x_reconst, x_quant) / 0.693147
    return loss
opt = optim.Adam(decoder.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.LambdaLR(opt, lambda step: 1-(step/STEPS))

def run_eval(step):
    if step % 10000 != 0:
        return
    losses = []
    dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
        transform=torchvision.transforms.ToTensor(), train=False)
    loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for x, y in loader:
                x, y = x.cuda(), y.cuda()
                x_reconst = decoder(quantize(x))
                x_quant = quantize(x)
                loss = F.cross_entropy(x_reconst, x_quant, reduction='none')
                loss = loss.mean(dim=[1,2,3])
                losses.append(loss)
    print('Test loss (BPC):', torch.cat(losses).mean().item() / 0.693147)

lib.utils.train_loop(forward, opt, STEPS, hook=run_eval, scheduler=scheduler)

run_eval(0)