"""
Warm-up: supervised CIFAR-10 model.
"""

import argparse
import lib.ops
import lib.utils
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim

BATCH_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--resnet_n', type=int, default=2)
parser.add_argument('--resnet_k', type=int, default=1)
args = parser.parse_args()
print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
    download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=4)
def _():
    while True:
        yield from loader
inf_loader = _()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = lib.ops.WideResnet(N=args.resnet_n, k=args.resnet_k)
        self.classifier = nn.Linear(64*args.resnet_k, 10)
    def forward(self, x):
        x = self.resnet(x)
        x = x.mean(dim=[2,3])
        x = self.classifier(x)
        return x
model = Model().cuda()

def forward():
    x, y = next(inf_loader)
    x, y = x.cuda(), y.cuda()
    x = (2*x) - 1 # Rescale to [-1, 1]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return loss
opt = optim.Adam(model.parameters(), lr=3e-4)

def extract_feats(train):
    dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
        transform=torchvision.transforms.ToTensor(), train=train)
    loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        Z, Y = [], []
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            x = (2*x) - 1 # Rescale to [-1, 1]
            _, z = model(x)
            Z.append(z)
            Y.append(y)
        Z = torch.cat(Z, dim=0)
        Y = torch.cat(Y, dim=0)
    return Z, Y

def run_eval():
    accs = []
    dataset = torchvision.datasets.CIFAR10(os.path.expanduser('~/data'),
        transform=torchvision.transforms.ToTensor(), train=False)
    loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            x = (2*x) - 1 # Rescale to [-1, 1]
            logits = model(x)
            acc = logits.argmax(dim=1).eq(y).float()
            accs.append(acc)
    print('Test acc:', torch.cat(accs).mean().item())

lib.utils.train_loop(forward, opt, 20*1000)
run_eval()