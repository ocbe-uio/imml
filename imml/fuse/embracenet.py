from .. import Module, deepmodule_installed

if deepmodule_installed:
    import torch


class EmbraceNet(Module):
    r"""
    PyTorch module to fuse modalities using EmbraceNet. [#embracenetpaper]_ [#embracenetcode1]_ [#embracenetcode2]_

    Parameters
    ----------
    missing_values : float, default=0.
        Value to use for missing data.

    References
    ----------
    .. [#embracenetpaper] Choi JH, Lee JS. EmbraceNet: A robust deep learning architecture for multimodal
                          classification. Information Fusion. 2019 Nov 1;51:259-70.
    .. [#embracenetcode1] https://github.com/luisvalesilva/multisurv/tree/master
    .. [#embracenetcode2] https://github.com/idearibosome/embracenet

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.fuse import EmbraceNet
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> fuse = EmbraceNet()
    >>> fuse(Xs)
    """

    def __init__(self, missing_values : float = None):
        super().__init__()
        if missing_values is None:
            missing_values = 0.
        self.missing_values = missing_values


    def _get_selection_probabilities(self, d, b):
        p = torch.ones(d.size(0), b)  # Size modalities x batch

        # Handle missing data
        for i, modality in enumerate(d):
            for j, batch_element in enumerate(modality):
                if len(torch.nonzero(batch_element)) < 1:
                    p[i, j] = self.missing_values

        # Equal chances to all available modalities in each mini batch element
        m_vector = torch.sum(p, dim=0)
        p /= m_vector

        return p


    def _get_sampling_indices(self, p, c, m):
        r = torch.multinomial(input=p.transpose(0, 1), num_samples=c, replacement=True)
        r = torch.nn.functional.one_hot(r.long(), num_classes=m)
        r = r.permute(2, 0, 1)

        return r


    def forward(self, x):
        x = torch.stack(x)
        m, b, c = x.size()

        p = self._get_selection_probabilities(x, b)
        r = self._get_sampling_indices(p, c, m).float().to(x.device)

        d_prime = r * x
        e = d_prime.sum(dim=0)

        return e