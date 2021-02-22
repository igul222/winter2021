"""
Train a 4096-dim LSTM as a character-level LM and finetune it as a sentiment
classifier.
"""

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
parser.add_argument('--load_weights_path', type=str, default=None)
# ~/jobs/2021_02_18_215248_04_big_noNgram
# ~/jobs/2021_02_18_215115_04_big_ngram
parser.add_argument('--sst_train_size', type=int, default=None)
parser.add_argument('--weight_decay', action='store_true')
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

    if args.load_weights_path is None:
        lib.utils.train_loop(forward, opt, steps=NGRAM_STEPS,
            print_freq=PRINT_FREQ)
        print('Saving ngram model weights...')
        torch.save(ngram_model.state_dict(), 'ngram_model.pt')
    else:
        weights = torch.load(os.path.join(args.load_weights_path, 'ngram_model.pt'))
        ngram_model.load_state_dict(weights)

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
        return logits, h_t.detach(), c_t.detach(), all_h

lstm = LSTM().cuda()
opt = optim.Adam(lstm.parameters(), lr=5e-4)
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
    logits, h0, c0, _ = lstm(inputs, h0, c0)
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

if args.load_weights_path is None:
    lib.utils.train_loop(forward, opt, steps=LSTM_STEPS,
        print_freq=PRINT_FREQ)
    print('Saving LSTM weights...')
    torch.save(lstm.state_dict(), 'lstm.pt')
else:
    weights = torch.load(os.path.join(args.load_weights_path, 'lstm.pt'))
    lstm.load_state_dict(weights)


def load_sst(split, n=None):
    # Load from disk
    path = os.path.join(SST_PATH, f'{split}_binary_sent.csv')
    X, y = [], []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                continue
            label, text = line[0], line[2:]
            if text[-1] == '\n':
                text = text[:-1]
            text = bytes_to_tensor(text.encode('utf-8'))
            X.append(text)
            y.append(int(label))

    # Subsample the dataset
    if n is not None:
        idx = list(torch.randperm(len(X))[:n])
        X = [x for i,x in enumerate(X) if i in idx]
        y = [y_ for i, y_ in enumerate(y) if i in idx]

    lengths = torch.tensor([len(x) for x in X], device='cuda')
    X = nn.utils.rnn.pad_sequence(X, batch_first=True).cuda()
    y = torch.tensor(y).cuda()

    return X, lengths, y

def multiclass_accuracy(y_pred, y):
    return torch.argmax(y_pred, dim=-1).eq(y).float()

print('Loading SST:')
X_train, lengths_train, y_train = load_sst('train', args.sst_train_size)
X_dev, lengths_dev, y_dev = load_sst('dev')
X_test, lengths_test, y_test = load_sst('test')

print('Training SST model:')

classifier = nn.Linear(LSTM_DIM, 2).cuda()
def forward():
    idx = torch.randperm(X_train.shape[0])[:BATCH_SIZE]
    lengths, X, y = lengths_train[idx], X_train[idx], y_train[idx]
    X = X[:, :lengths.max()]
    h0 = torch.zeros([1, X.shape[0], LSTM_DIM]).cuda()
    c0 = torch.zeros([1, X.shape[0], LSTM_DIM]).cuda()
    _, _, _, all_h = lstm(X, h0, c0)
    logits = classifier(all_h[torch.arange(X.shape[0]), lengths-1])
    loss = F.cross_entropy(logits, y)
    acc = multiclass_accuracy(logits, y).mean()
    return loss, acc

def eval_fn():
    X, lengths, y = X_test, lengths_test, y_test
    accs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            X_ = X[i:i+BATCH_SIZE]
            lengths_ = lengths[i:i+BATCH_SIZE]
            y_ = y[i:i+BATCH_SIZE]
            X = X[:, :lengths.max()]
            h0 = torch.zeros([1, X_.shape[0], LSTM_DIM]).cuda()
            c0 = torch.zeros([1, X_.shape[0], LSTM_DIM]).cuda()
            _, _, _, all_h = lstm(X_, h0, c0)
            logits = classifier(all_h[torch.arange(X_.shape[0]), lengths_-1])
            acc = multiclass_accuracy(logits, y_).mean()
            accs.append(acc)
    print(f'Test acc:', torch.stack(accs).mean().item())

opt = optim.Adam(classifier.parameters(), lr=5e-4, weight_decay=0.01)
lib.utils.train_loop(forward, opt, 1000, ['acc'], print_freq=100)
eval_fn()

opt = optim.Adam([
    {'params': lstm.parameters(), 'lr':5e-4},
    {'params': classifier.parameters(), 'lr':5e-4}
])
lib.utils.train_loop(forward, opt, 2*X_train.shape[0]//BATCH_SIZE,
    ['acc'])
eval_fn()