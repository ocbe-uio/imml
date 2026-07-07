from .. import Module, Tensor, deepmodule_installed

if deepmodule_installed:
    import torch


class ConcatFusion(Module):
    r"""
    PyTorch module to fuse modalities by concatenating them.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.fuse import ConcatFusion
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> fuse = ConcatFusion()
    >>> fuse(Xs)
    """


    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        out = torch.cat(x, dim=1)
        return out