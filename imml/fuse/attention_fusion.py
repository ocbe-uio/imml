from .. import Module, Tensor, deepmodule_installed

if deepmodule_installed:
    import torch


class AttentionFusion(Module):
    r"""
    PyTorch module to fuse modalities using the attention mechanism. [#attentionpaper]_

    References
    ----------
    .. [#attentionpaper] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I.
                         Attention is all you need. Advances in neural information processing systems. 2017;30.


    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.fuse import AttentionFusion
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> fuse = AttentionFusion()
    >>> fuse(Xs)
    """


    def __init__(self, n_features: int, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_features, bias=bias)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(0)

    def forward(self, x: Tensor):
        x = torch.stack(x)
        x = self.linear(x)
        scores = self.tanh(x)
        weights = self.softmax(scores)
        out = torch.sum(x * weights, dim=0)
        return out
