import os
import numpy as np


def load_inputs(test_path):
    names_path = os.path.join(test_path, 'input_names.txt')
    with open(names_path, 'r') as f:
        names = [x.strip() for x in f.readlines()]

    mixture = np.load(os.path.join(test_path, 'mixture.npy'))
    
    names.remove('mixture')
    state_buffers = []
    for name in names:
        buf = np.load(os.path.join(test_path, f'{name}.npy'))
        state_buffers.append(buf)

    return mixture, names, state_buffers