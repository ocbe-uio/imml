# License: BSD-3-Clause

import os
import zipfile
from os.path import dirname
import numpy as np
from PIL import Image

from ._m3care import NMT_tran, MM_transformer_encoder, init_weights, PositionalEncoding, clones, \
    GraphConvolution, length_to_mask, guassian_kernel
from .. import deepmodule_installed, deepmodule_error, LightningModule, Module

if deepmodule_installed:
    import torch
    from torch import optim, nn
    from torchvision import models as models
    import torch.nn.functional as F
    import torchvision.transforms as transforms


class M3Care(LightningModule):
    r"""

    Missing Modalities in Multimodal healthcare data (M3Care). [#m3carepaper]_ [#m3carecode]_ [#m3carecode2]_

    M3Care is a multimodal classification framework that handles missing modalities by imputing latent
    task-relevant information using similar samples, based on a modality-adaptive similarity metric.
    It supports heterogeneous input types (e.g., tabular, text, vision).

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    input_dim : list of int, default=None
        A list specifying the input dimensions for each tabular modality.
    hidden_dim : int, default=128
        Hidden dimension size.
    embed_size : int, default=128
        Size of the shared embedding space where modalities are projected.
    modalities : list of str, default=None
        Names of the modalities. Options are "tabular", "text" and "image".
    vocab : list, default=None
        List with path to corpus file, maximum number of words in vocabulary, and freq_cutoff
        (if word occurs n < freq_cutoff times, drop the word). If you want to pass your own Vocab object, use just
        a list with one element [Vocab]. If None, ["test.de-en.en", 50000, 2] will be
        used (if applicable). [#m3carecode]_
    learning_rate : float, default=1e-4
        Learning rate for the optimizer.
    weight_decay : float, default=1e-4
        Weight decay used by the optimizer.
    output_dim : int, default=1
        Number of classes in your response variable. Typically 1 for binary classification.
    loss_fn : callable, default=None
        Loss function. If None, defaults to `nn.BCEWithLogitsLoss()` if output_dim == <=2, else `nn.CrossEntropyLoss()`.
    keep_prob : float, default=0.5
        Dropout keep probability used in MLP layers.
    extractors : list of nn.Module, default=None
        List of custom feature extractors for each modality. If None, defaults will be used.

    References
    ----------
    .. [#m3carepaper] Zhang, Chaohe, et al. "M3care: Learning with missing modalities in multimodal healthcare data."
                      Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022.
    .. [#m3carecode] https://github.com/choczhang/M3Care/
    .. [#m3carecode2] https://github.com/pcyin/pytorch_basic_nmt/tree/master

    See Also
    --------
    :class:`~imml.load.M3CareDataset`

    Example
    --------
    >>> from lightning import Trainer
    >>> import numpy as np
    >>> import pandas as pd
    >>> from torch.utils.data import DataLoader
    >>> from imml.classify import M3Care
    >>> from imml.load import M3CareDataset
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((2, 10)))]
    >>> Xs.append(pd.DataFrame(np.random.default_rng(42).random((2, 15))))
    >>> Xs.append(pd.DataFrame(["docs/figures/graph.png", "docs/figures/logo_imml.png"]))
    >>> Xs.append(pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]))
    >>> Xs = Amputer(p=0.2, random_state=42).fit_transform(Xs) # this step is optional
    >>> y = pd.Series(np.random.default_rng(42).integers(0, 2, len(Xs[0])), dtype=np.float32)
    >>> train_data = M3CareDataset(Xs=Xs, y=y)
    >>> train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    >>> trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    >>> modalities = ["tabular", "tabular", "image", "text"]
    >>> estimator = M3Care(modalities=modalities, input_dim=[X.shape[1] for X,mod in zip(Xs, modalities) if mod=="tabular"])
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """

    def __init__(self, input_dim: list = None, hidden_dim: int = 128, embed_size: int = 128, modalities: list = None,
                 vocab: list = None, learning_rate: float = 1e-4, weight_decay: float = 1e-4, output_dim: int = 1,
                 loss_fn: callable = None, keep_prob: float = 0.5, extractors: list = None):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if input_dim is not None and not isinstance(input_dim, list):
            raise ValueError(f"Invalid input_dim. It must be a list. A {type(input_dim)} was passed.")
        if not isinstance(hidden_dim, int):
            raise ValueError(f"Invalid hidden_dim. It must be an integer. A {type(hidden_dim)} was passed.")
        if hidden_dim <= 0:
            raise ValueError(f"Invalid hidden_dim. It must be positive. {hidden_dim} was passed.")
        if not isinstance(embed_size, int):
            raise ValueError(f"Invalid embed_size. It must be an integer. A {type(embed_size)} was passed.")
        if embed_size <= 0:
            raise ValueError(f"Invalid embed_size. It must be positive. {embed_size} was passed.")
        if not isinstance(modalities, list):
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if len(modalities) < 2:
            raise ValueError(f"Invalid modalities. It must have at least two modalities. Got {len(modalities)} modalities")
        modalities_options = ["tabular", "text", "image"]
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
        if loss_fn is not None and not callable(loss_fn):
            raise ValueError(f"Invalid loss_fn. It must be callable. A {type(loss_fn)} was passed.")
        if not isinstance(keep_prob, float):
            raise ValueError(f"Invalid keep_prob. It must be a float. A {type(keep_prob)} was passed.")
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(f"Invalid keep_prob. It must be between 0 and 1. {keep_prob} was passed.")
        if extractors is not None and not isinstance(extractors, list):
            raise ValueError(f"Invalid extractors. It must be a list. A {type(extractors)} was passed.")
        if vocab is None:
            vocab_folder = dirname(__file__)
            vocab_filename = "test.de-en.en"
            vocab_folder = os.path.join(vocab_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            vocab_path = os.path.join(vocab_folder, vocab_filename)
            if not os.path.exists(vocab_path):
                download_vocab = input(
                    "You have not provided a vocab. If you want to use the default vocab, could you allow the download? Enter a bool"
                )
                if not bool(download_vocab):
                    raise ValueError("You have not provided a vocab. Please provide a vocab or allow the download.")
                try:
                    os.system("wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip")
                    with zipfile.ZipFile("iwslt2014_ende.zip", 'r') as zip_ref:
                        file_path = zip_ref.getinfo(os.path.join("data", vocab_filename))
                        file_path.filename = os.path.basename(file_path.filename)
                        zip_ref.extract(file_path, vocab_folder)
                except:
                    raise
                finally:
                    os.remove("iwslt2014_ende.zip")
            vocab = [vocab_path, 50000, 2]
        elif not isinstance(vocab, list):
            raise ValueError(f"Invalid vocab. It must be a list. A {type(vocab)} was passed.")

        super().__init__()

        self.model = M3CareModule(input_dim=input_dim, hidden_dim=hidden_dim, embed_size=embed_size, vocab=vocab,
                                  modalities=modalities, output_dim=output_dim, keep_prob=keep_prob, extractors=extractors)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
        self.loss_fn = loss_fn
        if output_dim == 1:
            self.get_probs = nn.Sigmoid()
        else:
            self.get_probs = nn.Softmax(dim=-1)


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, sum_of_diff = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = self.loss_fn(y_pred, y) + sum_of_diff.sum() * 1e-7
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = self.loss_fn(y_pred, y)
        return loss


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = self.loss_fn(y_pred, y)
        return loss


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        y_pred = self.get_probs(y_pred)
        return y_pred


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


class M3CareModule(Module):

    def __init__(self, input_dim: list = None, hidden_dim: int = 128, embed_size: int = 128, modalities: list = None,
                 vocab = None, output_dim: int =1, keep_prob: float = 1, extractors: list = None):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.modalities = modalities
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.n_mods = len(modalities)

        if extractors is None:
            extractors = [None] * len(modalities)
        if input_dim is not None:
            self.input_dim = iter(input_dim)

        for i, (mod, extractor) in enumerate(zip(self.modalities, extractors)):
            if mod == "tabular":
                if extractor is None:
                    extractor = nn.Linear(next(self.input_dim), hidden_dim)
            elif mod == "text":
                if extractor is None:
                    extractor = NMT_tran(embed_size=embed_size, hidden_size=hidden_dim,
                                         dropout_rate=1 - self.keep_prob, vocab=vocab)
            elif mod == "image":
                if extractor is None:
                    self.preprocess_img = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                    ])
                    extractor = nn.Sequential(models.resnet18(),
                                              nn.Linear(1000, self.hidden_dim)
                                              )
            setattr(self, f"extractor{i}", extractor)

        self.MM_model1 = MM_transformer_encoder(input_dim=self.hidden_dim, d_model=self.hidden_dim, \
                                               MHD_num_head=4, d_ff=self.hidden_dim * 4, output_dim=1)
        self.MM_model2 = MM_transformer_encoder(input_dim=self.hidden_dim, d_model=self.hidden_dim, \
                                                MHD_num_head=1, d_ff=self.hidden_dim * 4, output_dim=1)

        self.token_type_embeddings = nn.Embedding(6, self.hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.PositionalEncoding = PositionalEncoding(self.hidden_dim, dropout=0, max_len=5000)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)

        self.proj1 = nn.Linear(self.hidden_dim * len(self.modalities), self.hidden_dim * 2)
        self.out_layer = nn.Linear(self.hidden_dim * 2, self.output_dim)

        self.threshold = nn.Parameter(torch.ones(size=(1,)) + 1)
        self.simiProj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.bn = nn.BatchNorm1d(self.hidden_dim)

        self.simiProj = clones(torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        ), self.n_mods)

        self.GCN1 = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.n_mods)
        self.GCN2 = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.n_mods)
        self.weight1 = clones(nn.Linear(self.hidden_dim, 1), self.n_mods)
        self.weight2 = clones(nn.Linear(self.hidden_dim, 1), self.n_mods)
        self.eps = nn.ParameterList([nn.Parameter(torch.ones(1)+1) for _ in range(self.n_mods)])


    def forward(self, Xs, observed_mod_indicator):
        feats = []
        hidden00 = []
        mask_mats = []
        mask_mats_ = []
        mask2_mats = []
        for X_idx, (X,mod) in enumerate(zip(Xs, self.modalities)):
            extractor = getattr(self, f"extractor{X_idx}")
            if mod == 'tabular':
                X = X.clone()
                X[X.isnan().all(axis=1)] = 0
                feat = extractor(X)
                feat = F.relu(feat)
                if len(X) == 1:
                    mask = torch.ones((2, 1)).int().squeeze()[:1]
                else:
                    mask = torch.ones((feat.shape[0], 1)).int().squeeze()
                feat_00 = feat.clone()
            elif mod == 'image':
                X = [self.preprocess_img(Image.open(img_path).convert('RGB')
                     if bool(img_path) else Image.new("RGB", (256, 256), (0, 0, 0)))
                     for img_path in X]
                device = next(extractor[0].parameters()).device
                X = torch.stack(X).to(device)
                feat = extractor(X)
                feat = F.relu(feat)
                if len(X) == 1:
                    mask = torch.ones((2, 1)).int().squeeze()[:1]
                else:
                    mask = torch.ones((feat.shape[0], 1)).int().squeeze()
                feat_00 = feat.clone()
            elif mod == 'text':
                X = [f"[CLS] {s}".split() if bool(s) else "[CLS] None".split() for s in X]
                feat, lens = extractor(X)
                feat = F.relu(feat)
                mask = torch.from_numpy(np.array(lens))
                feat_00 = feat[:,0].clone()
            else:
                raise ValueError(f"Unknown modality type: {mod}")
            mask = length_to_mask(mask).unsqueeze(1).to(feat.device).int()
            mask_ = observed_mod_indicator[:, [X_idx]]
            mask2 = mask_ * mask_.permute(1,0)
            feats.append(feat)
            hidden00.append(feat_00)
            mask_mats.append(mask)
            mask_mats_.append(mask_)
            mask2_mats.append(mask2)

        sim_mats = []
        diffs = []
        for i, h in enumerate(hidden00):
            km1 = guassian_kernel(self.bn(F.relu(self.simiProj[i](h))), kernel_mul=2.0, kernel_num=3)
            km2 = guassian_kernel(self.bn(h), kernel_mul=2.0, kernel_num=3)
            sim = ((1 - torch.sigmoid(self.eps[i])) * km1 + torch.sigmoid(self.eps[i])) * km2
            if self.modalities[i] == "text":
                sim = sim * mask2_mats[i]
            sim_mats.append(sim)
            diff = torch.abs(torch.norm(self.simiProj[i](h), dim=1) - torch.norm(h, dim=1))
            diffs.append(diff)

        sum_of_diff = torch.stack(diffs, dim=1).sum(dim=1)

        sim_sum = torch.stack(sim_mats, dim=0).sum(dim=0)
        mask_sum = torch.stack(mask2_mats, dim=0).sum(dim=0).float()
        # Avoid division by zero: when no modality is shared between a pair (i,j),
        # mask_sum[i,j]=0 which would produce Inf, then Inf*0=NaN in GCN adjacency.
        similar_score = torch.where(mask_sum > 0, sim_sum / mask_sum.clamp(min=1), torch.zeros_like(sim_sum))

        th = torch.sigmoid(self.threshold)[0]
        similar_score = F.relu(similar_score - th)
        bin_mask = similar_score > 0
        similar_score = similar_score + bin_mask * th.detach()

        final_h = []
        gs = []
        for i, (h,mask2) in enumerate(zip(hidden00, mask2_mats)):
            g = F.relu(self.GCN1[i](similar_score*mask2, h))
            g = F.relu(self.GCN2[i](similar_score*mask2, g))
            gs.append(g)
            w1 = torch.sigmoid(self.weight1[i](g))
            w2 = torch.sigmoid(self.weight2[i](h))
            w1 = w1 / (w1 + w2)
            w2 = 1 - w1
            final = w1 * g + w2 * h
            final_h.append(final)

        embs = []
        for X_idx, (h,mod,final,g,mask,mask_) in enumerate(zip(feats, self.modalities, final_h, gs, mask_mats, mask_mats_)):
            h_ = torch.zeros_like(h)
            h_ = h_ + h
            if mod == "text":
                h_[mask_[:, 0], 0] = final[mask_[:, 0]]
                h_[torch.logical_not(mask_[:, 0]), 0] = g[torch.logical_not(mask_[:, 0])]
                emb = self.PositionalEncoding(h_)
            else:
                # Observed samples: use original features (matching original paper design).
                # Missing samples: use GCN output for imputation, consistent with how
                # the original handles missing text modalities.
                h_[torch.logical_not(mask_[:, 0])] = g[torch.logical_not(mask_[:, 0])]
                emb = h_.unsqueeze(1)
            mask = torch.ones_like(mask.permute(0,2,1).squeeze(-1)).to(h_.device).long()
            emb = emb + self.token_type_embeddings(X_idx * mask)
            embs.append(emb)

        z0 = torch.cat(embs, dim=1)
        z0_mask = torch.cat(mask_mats, dim=-1).int()
        z1 = F.relu(self.MM_model1(z0, z0_mask))
        z2 = F.relu(self.MM_model2(z1, z0_mask))

        # Track the starting position of each modality's first token in z0/z2.
        # Non-text modalities contribute 1 token; text contributes seq_len tokens.
        # Using X_idx directly as position is wrong when a text modality precedes others.
        positions = []
        pos = 0
        for emb in embs:
            positions.append(pos)
            pos += emb.shape[1]
        combined_hidden = [z2[:, positions[X_idx]] for X_idx in range(len(self.modalities))]
        combined_hidden = torch.cat(combined_hidden, dim=-1)
        last_hs_proj = self.dropout(F.relu(self.proj1(combined_hidden)))
        output = self.out_layer(last_hs_proj)
        output = output.squeeze(dim=1)

        return output, sum_of_diff
