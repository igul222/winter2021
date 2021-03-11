import collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import torch
from torch import optim

def print_tensor(label, tensor):
    """Print a tensor with a given label."""
    torch.set_printoptions(precision=3, linewidth=119, sci_mode=False)
    print(f'{label}:')
    for line in str(tensor).splitlines():
        print(f"\t{line}")
    torch.set_printoptions(profile='default')

def print_row(*row, colwidth=16):
    """Print a row of values."""
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str( x).ljust(colwidth)[:colwidth]
    print("  ".join([format_val(x) for x in row]))

def train_loop(forward, opt, steps, history_names=[], hook=None,
    print_freq=1000, scheduler=None, quiet=False):

    if not quiet:
        print_row('step', 'step time', 'loss', *history_names)
    histories = collections.defaultdict(lambda: [])
    scaler = torch.cuda.amp.GradScaler()
    start_time = time.time()
    for step in range(steps):

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            forward_vals = forward()
            if not isinstance(forward_vals, tuple):
                forward_vals = (forward_vals,)
        scaler.scale(forward_vals[0]).backward()
        scaler.step(opt)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if hook is not None:
            hook(step)

        histories['loss'].append(forward_vals[0].item())
        for name, val in zip(history_names, forward_vals[1:]):
            histories[name].append(val.item())

        if step % print_freq == 0:
            if not quiet:
                print_row(
                    step,
                    (time.time() - start_time) / (step+1),
                    np.mean(histories['loss']),
                    *[np.mean(histories[name]) for name in history_names]
                )
            histories.clear()