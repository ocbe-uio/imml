# License: BSD-3-Clause

from ._muse import FFNEncoder, RNNEncoder, TextEncoder, MML

try:
    from torch import optim, nn
    import lightning as L
    import torch.nn.functional as F
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

LightningModuleBase = L.LightningModule if deepmodule_installed else object
nnModuleBase = nn.Module if deepmodule_installed else object


class MUSE(LightningModuleBase):
    r"""

    MUtual-conSistEnt graph contrastive learning (MUSE). [#musepaper]_ [#musecode]_

    MUSE is a multimodal representation learning framework designed to handle missing modalities and partially
    labeled data. It uses a bipartite graph between samples and modalities to support arbitrary missingness patterns
    and a mutual-consistent contrastive loss to encourage the learning of label-discriminative, modality-consistent
    features.

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    input_dim : list of int, default=None
        A list specifying the input dimensions for each tabular/series modality.
    hidden_dim : int, default=128
        Hidden dimension size.
    modalities : list of str, default=None
        Names of the modalities. Options are "tabular", "text" and "series".
    tokenizer : str, default=None
        Tokenizer to use for text modality. If None, defaults to "emilyalsentzer/Bio_ClinicalBERT" tokenizer.
    learning_rate : float, default=2e-4
        Learning rate for the optimizer.
    weight_decay : float, default=0
        Weight decay used by the optimizer.
    output_dim : int, default=1
        Number of output dimensions. Typically 1 for binary classification.
    extractors : list of nn.Module, default=None
        List of custom feature extractors for each modality. If None, defaults will be used.
    gnn_layers : int, default=2
        Number of GNN layers used to propagate sample-modality representations.
    gnn_norm : str or None, default=None
        Optional normalization strategy in GNN layers (e.g., 'batchnorm', 'layernorm').
    code_pretrained_embedding : bool, default=True
        If True, initializes pretrained embeddings for text/code features.
    bert_type : str, default="prajjwal1/bert-tiny"
        HuggingFace model name or path for BERT backbone used in the text encoder.
    dropout : float, default=0.25
        Dropout rate applied in the encoders and classifier head.

    References
    ----------
    .. [#musepaper] Wu, Zhenbang, et al. "Multimodal patient representation learning with missing modalities and
                    labels." The Twelfth International Conference on Learning Representations. 2024.
    .. [#musecode] https://github.com/zzachw/MUSE/

    Example
    --------
    >>> from imml.classify import MUSE
    >>> from lightning import Trainer
    >>> import torch
    >>> import numpy as np
    >>> import pandas as pd
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import MUSEDataset
    >>> from imml.impute import get_observed_mod_indicator
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> train_data = MUSEDataset(Xs=[torch.from_numpy(X.values).float() for X in Xs],
                                 observed_mod_indicator=torch.from_numpy(get_observed_mod_indicator(Xs).values),
                                 y=torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float(),
                                 y_indicator=torch.ones((len(Xs[0]))).bool()
                                 )
    >>> train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = MUSE(modalities= ["tabular", "tabular"], input_dim=[X.shape[1] for X in Xs])
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """

    def __init__(self, input_dim: list = None, hidden_dim: int = 128, modalities: list = None,
                 tokenizer=None, learning_rate: float = 2e-4, weight_decay: float = 0., output_dim: int = 1,
                 extractors: list = None, gnn_layers: int = 2, gnn_norm: str = None,
                 code_pretrained_embedding: bool = True, bert_type: str = "prajjwal1/bert-tiny", dropout: float = 0.25):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if input_dim is not None and not isinstance(input_dim, list):
            raise ValueError(f"Invalid input_dim. It must be a list. A {type(input_dim)} was passed.")
        if not isinstance(hidden_dim, int):
            raise ValueError(f"Invalid hidden_dim. It must be an integer. A {type(hidden_dim)} was passed.")
        if hidden_dim <= 0:
            raise ValueError(f"Invalid hidden_dim. It must be positive. {hidden_dim} was passed.")
        if not isinstance(modalities, list):
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if len(modalities) < 1:
            raise ValueError(f"Invalid modalities. It must have at least one modality. Got {len(modalities)} modalities")
        modalities_options = ["tabular", "text", "series"]
        if not all(mod in modalities_options for mod in modalities):
            raise ValueError(f"Invalid modalities. Expected options are: {modalities_options}")
        if not isinstance(learning_rate, float):
            raise ValueError(f"Invalid learning_rate. It must be a float. A {type(learning_rate)} was passed.")
        if learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate. It must be positive. {learning_rate} was passed.")
        if not isinstance(weight_decay, float):
            raise ValueError(f"Invalid weight_decay. It must be a float. A {type(weight_decay)} was passed.")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay. It must be non-negative. {weight_decay} was passed.")
        if not isinstance(output_dim, int):
            raise ValueError(f"Invalid output_dim. It must be an integer. A {type(output_dim)} was passed.")
        if output_dim <= 0:
            raise ValueError(f"Invalid output_dim. It must be positive. {output_dim} was passed.")
        if extractors is not None and not isinstance(extractors, list):
            raise ValueError(f"Invalid extractors. It must be a list. A {type(extractors)} was passed.")
        if not isinstance(gnn_layers, int):
            raise ValueError(f"Invalid gnn_layers. It must be an integer. A {type(gnn_layers)} was passed.")
        if gnn_layers <= 0:
            raise ValueError(f"Invalid gnn_layers. It must be positive. {gnn_layers} was passed.")
        if gnn_norm is not None and not isinstance(gnn_norm, str):
            raise ValueError(f"Invalid gnn_norm. It must be a string. A {type(gnn_norm)} was passed.")
        if not isinstance(code_pretrained_embedding, bool):
            raise ValueError(f"Invalid code_pretrained_embedding. It must be a boolean. A {type(code_pretrained_embedding)} was passed.")
        if not isinstance(bert_type, str):
            raise ValueError(f"Invalid bert_type. It must be a string. A {type(bert_type)} was passed.")
        if not isinstance(dropout, float):
            raise ValueError(f"Invalid dropout. It must be a float. A {type(dropout)} was passed.")
        if dropout < 0 or dropout > 1:
            raise ValueError(f"Invalid dropout. It must be between 0 and 1. {dropout} was passed.")

        super().__init__()

        self.model = MUSEModule(input_dim=input_dim, tokenizer=tokenizer, hidden_dim=hidden_dim,
                               modalities=modalities, output_dim=output_dim, extractors=extractors,
                               gnn_layers=gnn_layers, gnn_norm=gnn_norm, bert_type=bert_type, dropout=dropout,
                               code_pretrained_embedding=code_pretrained_embedding)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        pred = self.model.predict(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        return pred


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


class MUSEModule(nnModuleBase):

    def __init__(self, input_dim: list = None, modalities: list = None, extractors: list = None, tokenizer=None,
                 hidden_dim: int = 128, output_dim: int = 1, gnn_layers: int = 2, gnn_norm: str = None,
                 code_pretrained_embedding: bool = True, bert_type: str = "prajjwal1/bert-tiny", dropout: float = 0.25
                 ):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()

        self.tokenizer = tokenizer
        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.code_pretrained_embedding = code_pretrained_embedding
        self.bert_type = bert_type
        self.dropout = dropout
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm

        self.dropout_layer = nn.Dropout(dropout)

        if modalities is None:
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if extractors is None:
            extractors = [None] * len(modalities)
        if input_dim is not None:
            self.input_dim = iter(input_dim)

        for i, (mod, extractor) in enumerate(zip(self.modalities, extractors)):
            if mod == "tabular":
                if extractor is None:
                    encoder = FFNEncoder(input_dim=next(self.input_dim), hidden_dim=hidden_dim,
                                         output_dim=hidden_dim, dropout_prob=dropout, num_layers=2)
                    extractor = nn.Sequential(encoder, nn.Linear(hidden_dim, hidden_dim))
            elif mod == "series":
                if extractor is None:
                    encoder = RNNEncoder(input_size=next(self.input_dim), hidden_size=hidden_dim, num_layers=1,
                                         rnn_type="GRU", dropout=dropout, bidirectional=False)
                    extractor = nn.Sequential(encoder, nn.Linear(hidden_dim, hidden_dim))
            elif mod == "text":
                if extractors is None:
                    encoder = TextEncoder(bert_type)
                    for param in encoder.parameters():
                        param.requires_grad = False
                    output_dim = encoder.model.config.hidden_size
                    extractor = nn.Sequential(encoder, nn.Linear(output_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown modality type: {mod}")
            setattr(self, f"extractor{i}", extractor)

        self.mml = MML(num_modalities=len(modalities), hidden_channels=hidden_dim, num_layers=gnn_layers,
                       dropout=dropout, normalize_embs=gnn_norm, output_dim=output_dim)


    def forward(self, Xs, y, observed_mod_indicator, y_indicator):
        observed_mod_indicator = ~observed_mod_indicator
        transformed_Xs = []
        for X_idx, (X,mod) in enumerate(zip(Xs, self.modalities)):
            extractor = getattr(self, f"extractor{X_idx}")
            code_embedding = extractor(X)
            code_embedding[observed_mod_indicator[:,X_idx]] = 0
            code_embedding = self.dropout_layer(code_embedding)
            transformed_Xs.append(code_embedding)
        loss = self.mml(Xs=transformed_Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def predict(self, Xs, observed_mod_indicator):
        observed_mod_indicator = ~observed_mod_indicator
        transformed_Xs = []
        for X_idx, (X,mod) in enumerate(zip(Xs, self.modalities)):
            extractor = getattr(self, f"extractor{X_idx}")
            code_embedding = extractor(X)
            code_embedding[observed_mod_indicator[:,X_idx]] = 0
            code_embedding = self.dropout_layer(code_embedding)
            transformed_Xs.append(code_embedding)
        logits = self.mml.inference(Xs=transformed_Xs, observed_mod_indicator=observed_mod_indicator)[0]
        return logits
