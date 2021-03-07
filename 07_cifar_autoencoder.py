"""
Semi-supervised classification from autoencoder representations on CIFAR-10.
"""

import lib.utils
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim

BATCH_SIZE = 128
Z_DIM = 128

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
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(256, 128, 5, padding=2, stride=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 64, 5, padding=2, stride=2, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(64, 3, 5, padding=2, stride=2, output_padding=1)
        self.linear1 = nn.Linear(4*4*256, 128)
        self.linear2 = nn.Linear(128, 4*4*256)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        z = self.linear1(x.view(x.shape[0], 4*4*256))
        z_pre = z

        z = z * 11 / z.norm(p=2, dim=1, keepdim=True)
        z = z + (torch.randn_like(z) * 1e-0)

        x = self.linear2(z).view(x.shape[0], 256, 4, 4)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        return x, z_pre

model = Model().cuda()

def forward():
    x, y = next(inf_loader)
    x, y = x.cuda(), y.cuda()
    x = (2*x) - 1 # Rescale to [-1, 1]
    x_reconst, _ = model(x)
    loss = (x - x_reconst).pow(2).mean()
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

def run_eval(_=None):
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

run_eval()
lib.utils.train_loop(forward, opt, 20*1000, hook=run_eval)