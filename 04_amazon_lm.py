import os
import torch

DATA_PATH = os.path.expanduser('~/data/amazon_reviews/all_reviews.txt')

data_size = os.path.getsize(DATA_PATH)
def read_bytes(n):
    """Read a random n-byte sequence of data."""
    offset = torch.randint(low=0, high=data_size-n+1, size=[]).item()
    with open(DATA_PATH, 'r') as f:
        f.seek(offset)
        return f.read(n)

for _ in range(100):
    print(read_bytes(100))