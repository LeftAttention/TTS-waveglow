import numpy as np
from scipy.io.wavfile import read
import torch
import os

import argparse
import json

class ParseFromConfigFile(argparse.Action):

    def __init__(self, option_strings, type, dest, help=None, required=False):
        super(ParseFromConfigFile, self).__init__(option_strings=option_strings, type=type, dest=dest, help=help, required=required)

    def __call__(self, parser, namespace, values, option_string):
        with open(values, 'r') as f:
            data = json.load(f)

        for group in data.keys():
            for k,v in data[group].items():
                underscore_k = k.replace('-', '_')
                setattr(namespace, underscore_k, v)

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            path = os.path.join(root, parts[0])
            text = parts[1]
            return path,text
        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x
