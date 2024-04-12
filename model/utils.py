"""
Utilites for training and evaluating models.

Code taken from https://github.com/openai/improved-diffusion, and modifed by Peter Zdravecký.
"""

import io
import blobfile as bf
import torch as th

def dev():
    """
    Get the device to use for computation.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)
