import argparse
import lib.utils
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm


# Run 03_amazon_preprocess.py to generate this file.
AMAZON_PATH = os.path.expanduser('~/data/amazon_reviews/all_reviews.txt')

# Download the CSV files from https://github.com/openai/
# generating-reviews-discovering-sentiment/tree/master/data
# into this directory:
SST_PATH = os.path.expanduser('~/data/SST')

# Data loader settings
BATCH_SIZE = 128
CHUNK_LEN = 256
N_CHUNKS = 20

# n-gram model settings
NGRAM_N = 3
NGRAM_DIM = 2048
NGRAM_STEPS = 10*1000

# LSTM settings
LSTM_DIM = 4096
LSTM_STEPS = 200*1000

# Misc.
PRINT_FREQ = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--ngram_enabled', action='store_true')
args = parser.parse_args()
print('Args:')
for k,v in sorted(vars(args).items()):
    print(f'\t{k}: {v}')

def bytes_to_tensor(x):
    """Convert a python bytes object to a pytorch uint8 tensor."""
    result = x
    result = np.frombuffer(result, dtype='uint8')
    result = np.copy(result)
    result = torch.tensor(result).cuda().long()
    return result

data_size = os.path.getsize(AMAZON_PATH)
def read_bytes(n):
    """Read a random n-byte sequence of data."""
    offset = torch.randint(low=0, high=data_size-n+1, size=[]).item()
    with open(AMAZON_PATH, 'r') as f:
        f.seek(offset)
        result = f.read(n)
    result = bytes_to_tensor(result.encode('utf-8'))
    return result

def data_generator():
    """
    A generator which infinitely draws a batch of long sequences, splits them
    into shorter chunks, and yields those chunks.
    """
    while True:
        buffer = read_bytes(BATCH_SIZE * CHUNK_LEN * N_CHUNKS)
        buffer = buffer.view(BATCH_SIZE, N_CHUNKS, CHUNK_LEN)
        for i in range(N_CHUNKS):
            yield buffer[:, i, :]
data_iterator = data_generator()

if args.ngram_enabled:
    print('Training n-gram model:')

    class NgramModel(nn.Module):
        def __init__(self):
            super(NgramModel, self).__init__()
            self.embedding = nn.Embedding(256, 256)
            self.conv1 = nn.Conv1d(256, NGRAM_DIM, NGRAM_N)
            self.conv2 = nn.Conv1d(NGRAM_DIM, NGRAM_DIM, 1)
            self.conv3 = nn.Conv1d(NGRAM_DIM, 256, 1)
        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(0,2,1)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)        
            x = x.permute(0,2,1)
            return x

    ngram_model = NgramModel().cuda()
    opt = optim.Adam(ngram_model.parameters(), lr=5e-4)
    def forward():
        inputs = next(data_iterator)
        logits = ngram_model(inputs)
        loss = F.cross_entropy(
            logits[:,:-1,:].reshape(-1, 256),
            inputs[:,NGRAM_N:].reshape(-1)
        )
        return loss
    lib.utils.train_loop(forward, opt, steps=NGRAM_STEPS, print_freq=PRINT_FREQ)

    print('Saving ngram model weights...')
    torch.save(ngram_model.state_dict(), 'ngram_model.pt')

print('Training LSTM:')

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=LSTM_DIM,
            num_layers=1,
            batch_first=True
        )
        self.lstm = nn.utils.weight_norm(self.lstm, 'weight_ih_l0')
        self.lstm = nn.utils.weight_norm(self.lstm, 'weight_hh_l0')
        self.readout = nn.Linear(LSTM_DIM, 256)

    def forward(self, x, h0, c0):
        x = self.embedding(x)
        all_h, (h_t, c_t) = self.lstm(x, (h0, c0))
        logits = self.readout(all_h)
        return logits, h_t.detach(), c_t.detach()

model = LSTM().cuda()
opt = optim.Adam(model.parameters(), lr=5e-4)
h0 = torch.zeros([1, BATCH_SIZE, LSTM_DIM]).cuda()
c0 = torch.zeros([1, BATCH_SIZE, LSTM_DIM]).cuda()
step = 0
def forward():
    global h0, c0, step
    inputs = next(data_iterator)
    if step % N_CHUNKS == 0:
        h0.zero_()
        c0.zero_()
    step += 1
    logits, h0, c0 = model(inputs, h0, c0)
    logits = logits[:, NGRAM_N-1:-1, :]
    targets = inputs[:, NGRAM_N:]
    if args.ngram_enabled:
        with torch.no_grad():
            ngram_logits = ngram_model(inputs)
        logits = logits + ngram_logits[:, :-1, :]
    loss = F.cross_entropy(
        logits.reshape(-1, 256),
        targets.reshape(-1))
    return loss
lib.utils.train_loop(forward, opt, steps=LSTM_STEPS, print_freq=PRINT_FREQ)

print('Saving LSTM weights...')
torch.save(model.state_dict(), 'lstm.pt')

def load_sst(split):
    path = os.path.join(SST_PATH, f'{split}_binary_sent.csv')
    X, y = [], []
    with open(path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            if i==0:
                continue
            label, text = line[0], line[2:]
            if text[-1] == '\n':
                text = text[:-1]
            text = bytes_to_tensor(text.encode('utf-8'))
            with torch.no_grad():
                h0 = torch.zeros([1, 1, LSTM_DIM]).cuda()
                c0 = torch.zeros([1, 1, LSTM_DIM]).cuda()
                _, ht, ct = model(text[None, :], h0, c0)
                feats = ht.view(-1)
            X.append(feats)
            y.append(int(label))
    return torch.stack(X, dim=0).cuda(), torch.tensor(y).cuda()

def multiclass_accuracy(y_pred, y):
    return torch.argmax(y_pred, dim=-1).eq(y).float()

print('Loading SST:')
X_train, y_train = load_sst('train')
X_dev, y_dev = load_sst('dev')

print('Training SST model:')
model = nn.Linear(X_train.shape[1], 2).cuda()
def forward():
    logits = model(X_train)
    loss = F.cross_entropy(logits, y_train)
    acc = multiclass_accuracy(logits, y_train).mean()
    return loss, acc
opt = optim.Adam(model.parameters())
lib.utils.train_loop(forward, opt, 1000, ['acc'], print_freq=100)

dev_acc = multiclass_accuracy(model(X_dev), y_dev).mean().item()
print('SST dev acc:', dev_acc)