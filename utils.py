import torch
import math
import os
import re
import csv
import sys


class CSVLogger(object):
    def __init__(self, filename, args, keys):
        self.filename = filename
        self.args = args
        self.keys = keys
        self.values = {k:[] for k in keys}
        self.init_file()
        
    def init_file(self):
        # This will overwrite previous file
        if os.path.exists(self.filename):
            return
        
        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.filename, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
#             logwriter.writerow([str(self.args)])        
            logwriter.writerow(self.keys)        
        
    def write_row(self, values):
        assert len(values) == len(self.keys)
        if not os.path.exists(self.filename):
            self.init_file()
        with open(self.filename, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(values)        
        

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def get_model_grads(model):
    return [p.grad.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]

def get_model_params(model):
    return [p.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]


def norm_diff(list1, list2=None):
    if not list2:
        list2 = [0] * len(list1)
    assert len(list1) == len(list2)
    return math.sqrt(sum((list1[i]-list2[i]).norm()**2 for i in range(len(list1))))