from .. import Module, Tensor, deepmodule_installed

if deepmodule_installed:
    import torch


class SumFusion(Module):
    r"""
    PyTorch module to fuse modalities using the sum operation.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.fuse import SumFusion
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> fuse = SumFusion()
    >>> fuse(Xs)
    """

    def __init__(self):
        super().__init__()


    def forward(self, x: Tensor):
        out = torch.stack(x).nansum(axis=0)
        return out
