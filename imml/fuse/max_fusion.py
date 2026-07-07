from .. import Module, Tensor, deepmodule_installed

if deepmodule_installed:
    import torch


class MaxFusion(Module):
    r"""
    PyTorch module to fuse modalities using the max operation.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.fuse import MaxFusion
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> fuse = MaxFusion()
    >>> fuse(Xs)
    """

    def __init__(self):
        super().__init__()


    def forward(self, x: Tensor):
        out = torch.stack(x).max(axis=0)[0]
        return out
