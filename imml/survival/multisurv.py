from ._multisurv import FC, Loss
from .. import deepmodule_installed, LightningModule, Module, deepmodule_error
from ..fuse import MaxFusion

if deepmodule_installed:
    import torch
    from torch import nn, optim


class MultiSurv(LightningModule):
    r"""

    MULTImodal SURVival prediction (MultiSurv). [#multisurvpaper]_ [#multisurvcode]_

    MultiSurv is a multimodal discrete-time survival model. Each modality is mapped to a
    shared latent feature size, available modality representations are fused, and a fully
    connected block predicts conditional survival probabilities for a sequence of time
    intervals.

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    input_dim : list of int, default=None
        A list specifying the input dimensions for each default tabular modality extractor.
    fusion : torch.nn.Module, default=None
        Fusion module used when more than one modality is available. If None, :class:`~imml.fuse.MaxFusion` is used.
    t_binds : int, default=30
        Number of output survival intervals.
    hidden_dim : int, default=512
        Size of each modality representation, matching the original MultiSurv default.
    embed_size : int, default=512
        Size of the representation passed to the final risk layer.
    extractors : list of torch.nn.Module, default=None
        Custom feature extractors for each modality. If None, fully connected tabular
        extractors equivalent to the original omics submodels are used.
    n_layers : int, default=4
        Number of layers in the post-fusion fully connected block.
    time_points : array-like, default=None
        Interval break points in days. If None, yearly breaks from 0 to 30 years are used.
    learning_rate : float, default=1e-4
        Learning rate for the optimizer.
    weight_decay : float, default=1e-4
        Weight decay used by the optimizer.

    References
    ----------
    .. [#multisurvpaper] Vale-Silva, Luis A., and Karl Rohr. "Long-term cancer survival prediction using
                          multimodal deep learning." Scientific Reports 11, 13505 (2021).
    .. [#multisurvcode] https://github.com/luisvalesilva/multisurv

    See Also
    --------
    :class:`~imml.load.MultiSurvDataset`

    Example
    --------
    >>> from lightning import Trainer
    >>> import numpy as np
    >>> import pandas as pd
    >>> from torch.utils.data import DataLoader
    >>> from imml.survival import MultiSurv
    >>> from imml.load import MultiSurvDataset
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((8, 10))) for _ in range(2)]
    >>> y = pd.DataFrame({"time": np.random.default_rng(42).uniform(0.5, 5, 8),
    ...                   "event": np.random.default_rng(42).integers(0, 2, 8)})
    >>> train_data = MultiSurvDataset(Xs=Xs, y=y)
    >>> train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    >>> trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    >>> estimator = MultiSurv(input_dim=[10, 10])
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """

    def __init__(self, input_dim: list = None, fusion: Module = None, t_binds: int = 30,
                 hidden_dim: int = 512, embed_size: int = 512, extractors: list = None, n_layers : int = 4,
                 time_points: list = None, learning_rate: float = 1e-4, weight_decay: float = 1e-4):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if input_dim is not None and not isinstance(input_dim, list):
            raise ValueError(f"Invalid input_dim. It must be a list. A {type(input_dim)} was passed.")
        if not isinstance(t_binds, int):
            raise ValueError(f"Invalid t_binds. It must be an integer. A {type(t_binds)} was passed.")
        if t_binds <= 0:
            raise ValueError(f"Invalid t_binds. It must be positive. {t_binds} was passed.")
        if not isinstance(embed_size, int):
            raise ValueError(f"Invalid embed_size. It must be an integer. A {type(embed_size)} was passed.")
        if embed_size <= 0:
            raise ValueError(f"Invalid embed_size. It must be positive. {embed_size} was passed.")
        if not isinstance(hidden_dim, int):
            raise ValueError(f"Invalid hidden_dim. It must be an integer. A {type(hidden_dim)} was passed.")
        if hidden_dim <= 0:
            raise ValueError(f"Invalid hidden_dim. It must be positive. {hidden_dim} was passed.")
        if not isinstance(n_layers, int):
            raise ValueError(f"Invalid n_layers. It must be an integer. A {type(n_layers)} was passed.")
        if n_layers <= 0:
            raise ValueError(f"Invalid n_layers. It must be positive. {n_layers} was passed.")
        if not isinstance(learning_rate, float):
            raise ValueError(f"Invalid learning_rate. It must be a float. A {type(learning_rate)} was passed.")
        if learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate. It must be positive. {learning_rate} was passed.")
        if not isinstance(weight_decay, float):
            raise ValueError(f"Invalid weight_decay. It must be a float. A {type(weight_decay)} was passed.")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay. It must be non-negative. {weight_decay} was passed.")
        if fusion is not None and not isinstance(fusion, Module):
            raise ValueError(f"Invalid fusion. It must be a PyTorch Module. A {type(fusion)} was passed.")
        if extractors is not None and not isinstance(extractors, list):
            raise ValueError(f"Invalid extractors. It must be a list. A {type(extractors)} was passed.")
        if extractors is not None and not all(isinstance(extractor, Module) for extractor in extractors):
            raise ValueError(f"Invalid extractors. All extractors must be PyTorch Modules. Got {type(extractors[0])}")
        if extractors is None:
            if input_dim is None:
                raise ValueError("Invalid input_dim. It must be provided when extractors is None.")

        super().__init__()

        if fusion is None:
            fusion = MaxFusion()
        if extractors is None:
            extractors = [FC(in_features=n, out_features=hidden_dim, n_layers=3) for n in input_dim]
        if time_points is None:
            time_points = torch.arange(0., 365. * (t_binds + 1), 365.)
        elif not torch.is_tensor(time_points):
            time_points = torch.as_tensor(time_points, dtype=torch.float)
        else:
            time_points = time_points.float()
        if len(time_points) != t_binds + 1:
            raise ValueError(f"Invalid time_points. It must contain t_binds + 1 break points. Got {len(time_points)} break points for {t_binds} intervals.")

        self.model = MultiSurvModule(fusion=fusion, t_binds=t_binds, num_features=hidden_dim,
                                     embed_size=embed_size, n_layers=n_layers, extractors=extractors)
        self.time_points = time_points
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = Loss()


    def _forward(self, batch, batch_idx):
        Xs, y = batch
        event, time = y[:, 0], y[:, 1]
        feature_representations, risk = self.model(Xs)
        loss = self.loss_fn(risk=risk, times=time, events=event, breaks=self.time_points.to(risk.device))
        return loss, risk, time, event


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        loss = self._forward(batch=batch, batch_idx=batch_idx)[0]
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        loss = self._forward(batch=batch, batch_idx=batch_idx)[0]
        return loss


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        loss = self._forward(batch=batch, batch_idx=batch_idx)[0]
        return loss


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        loss, risk, time, event = self._forward(batch=batch, batch_idx=batch_idx)
        risk = risk.detach()
        y_pred = self._predictions_to_pycox(risk, time_points=None)
        return y_pred


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def _predictions_to_pycox(self, preds, time_points=None):
        y_pred = torch.cumprod(preds, dim=1)
        initial_survival = torch.ones((y_pred.shape[0], 1), dtype=y_pred.dtype, device=y_pred.device)
        y_pred = torch.cat((initial_survival, y_pred), dim=1)
        return y_pred.transpose(0, 1)


class MultiSurvModule(Module):

    def __init__(self, fusion, t_binds, num_features, embed_size, n_layers, extractors):

        super().__init__()

        self.fusion = fusion
        self.extractors = nn.ModuleList(extractors)
        self.fc_block = FC(in_features=num_features, out_features=embed_size, n_layers=n_layers)
        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=embed_size, out_features=t_binds),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        features = [extractor(modality_x) for modality_x, extractor in zip(x, self.extractors)]
        fused = self.fusion(features)

        x = self.fc_block(fused)
        risk = self.risk_layer(x)
        output_features = [X[~X.eq(0).all(axis=1)] for X in features]
        return output_features, risk
