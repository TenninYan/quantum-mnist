import numpy as np

def normalize(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    norm = (x-xmean)/6*xstd + 0.5
    clipped = np.clip(norm, 0.0, 1.0)
    return clipped
