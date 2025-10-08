# License: BSD-3-Clause

import numpy as np
import pandas as pd
import snf
from sklearn.cluster import spectral_clustering, SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.utils import check_symmetric
from snf.compute import _find_dominate_set

from ._integrao._aux_integrao import data_indexing, dist2, _stable_normalized_pd, _scaling_normalized_pd, p_preprocess, \
    _stable_normalized
from ..preprocessing import remove_missing_samples_by_mod

try:
    import torch
    import lightning as L
    from torch import nn, optim, autograd
    from torch_geometric.nn import GraphSAGE
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

LightningModuleBase = L.LightningModule if deepmodule_installed else object
nnModuleBase = nn.Module if deepmodule_installed else object


class IntegrAO(LightningModuleBase):
    r"""
    Integrate Any Omics (IntegrAO). [#integraopaper]_ [#integraocode]_

    IntegrAO first combines partially overlapping sample graphs from diverse sources and utilizes graph neural
    networks to produce unified sample embeddings.

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities. It will be used to create the neural network architecture.
    model : nn.Module, default=None
        Deep learning model. If None, it will select IntegrAOModule.
    n_clusters : int, default=8
        The number of clusters to generate.
    neighbor_size : int, default=None
        Number of neighbors to use. If None, it will use N/6).
    hidden_channels : int, default=128
        Hidden dimension size.
    embedding_dims : int, default=50
        Size of the shared embedding space where modalities are projected.
    fusing_iteration : int, default=20
        Number of iterations for fusing.
    mu : float, default=0.5
        Normalization factor to scale similarity kernel.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    weight_decay : float, default=2e-2
        Weight decay used by the optimizer.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.

    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_clusters)
        Commont latent feature matrix.
    cluster_model_ : SpectralClustering
        Scikit-learn SpectralClustering object.
    fused_networks_ : list of array-like of shape (n_samples_i, n_samples_i)
        Modal-specific graphs.

    References
    ----------
    .. [#integraopaper] Ma, Shihao, et al. "Moving towards genome-wide data integration for patient stratification
                        with Integrate Any Omics." Nature Machine Intelligence 7.1 (2025): 29-42.
    .. [#integraocode] https://github.com/bowang-lab/IntegrAO

    Example
    --------
    >>> import numpy as np
    >>> import torch
    >>> from imml.cluster import IntegrAO
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import IntegrAODataset
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = IntegrAO(Xs=Xs, random_state=42)
    >>> train_data = IntegrAODataset(Xs=Xs, neighbor_size=estimator.neighbor_size, networks=estimator.fused_networks_)
    >>> train_dataloader = DataLoader(dataset=train_data)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> trainer.fit(estimator, train_dataloader)
    >>> labels = trainer.predict(estimator, train_dataloader)[0]
    """

    def __init__(self, Xs, model : nnModuleBase = None, n_clusters: int = 8, neighbor_size : int = None,
                 hidden_channels : int = 128, embedding_dims: int = 50, fusing_iteration: int = 20,
                 mu : float = 0.5, learning_rate : float = 1e-3, weight_decay : float = 1e-4, random_state : int = None):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list of array-likes objects objects. A {type(Xs)} was passed.")
        if not isinstance(learning_rate, float):
            raise ValueError(f"Invalid learning_rate. It must be a float. A {type(learning_rate)} was passed.")
        if learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate. It must be a positive number. {learning_rate} was passed.")

        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        if neighbor_size is None:
            neighbor_size = int(Xs[0].shape[0]/6)
        self.neighbor_size = neighbor_size
        self.n_clusters = n_clusters
        self.embedding_dims = embedding_dims
        self.fusing_iteration = fusing_iteration
        self.mu = mu
        self.hidden_channels = hidden_channels
        self.loss_mse = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

        Xs = remove_missing_samples_by_mod(Xs=Xs)

        (
            self.dicts_common,
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dicts_unique,
            self.original_order,
            self.dict_original_order,
        ) = data_indexing(Xs)

        self._network_diffusion(Xs=Xs)
        if model is None:
            model = IntegrAOModule(in_channels=[X.shape[1] for X in Xs], hidden_channels=hidden_channels,
                                    out_channels=embedding_dims)
        self.model = model

        ps = []
        for network in self.fused_networks_:
            p = p_preprocess(network)
            p = torch.from_numpy(p).float()
            ps.append(p)
        self.ps = ps


    def forward(self, Xs, average=True):
        return self.model(Xs, average=average)


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """

        embeddings = self(Xs=batch, average=False)
        kl_loss = sum([self._tsne_loss(self.ps[i], X_embedding) for i,X_embedding in enumerate(embeddings)])
        alignment_loss = sum([self.loss_mse(
            embeddings[i][self.dicts_commonIndex[(i, j)]], embeddings[j][self.dicts_commonIndex[(j, i)]])
            for i in range(len(embeddings) - 1) for j in range(i + 1, len(embeddings))
        ])
        loss = kl_loss + alignment_loss
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        embeddings = self(Xs=batch, average=False)
        kl_loss = sum([self._tsne_loss(self.ps[i], X_embedding) for i,X_embedding in enumerate(embeddings)])
        alignment_loss = sum([self.loss_mse(
            embeddings[i][self.dicts_commonIndex[(i, j)]], embeddings[j][self.dicts_commonIndex[(j, i)]])
            for i in range(len(embeddings) - 1) for j in range(i + 1, len(embeddings))
        ])
        loss = kl_loss + alignment_loss
        return loss


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        embeddings = self(Xs=batch, average=False)
        kl_loss = sum([self._tsne_loss(self.ps[i], X_embedding) for i,X_embedding in enumerate(embeddings)])
        alignment_loss = sum([self.loss_mse(
            embeddings[i][self.dicts_commonIndex[(i, j)]], embeddings[j][self.dicts_commonIndex[(j, i)]])
            for i in range(len(embeddings) - 1) for j in range(i + 1, len(embeddings))
        ])
        loss = kl_loss + alignment_loss
        return loss


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        embeddings = self(Xs=batch)
        embeddings = pd.DataFrame(data=embeddings, index=self.dict_sampleToIndexs.keys()).sort_index().values
        dist_final = dist2(embeddings, embeddings)
        Wall_final = snf.compute.affinity_matrix(dist_final, K=self.neighbor_size, mu=self.mu)
        Wall_final = _stable_normalized(Wall_final)
        if getattr(self, "cluster_model_", None) is None:
            self.cluster_model_ = SpectralClustering(n_clusters=self.n_clusters, random_state=self.random_state,
                                                     affinity="precomputed")
        labels = self.cluster_model_.fit_predict(X=Wall_final)
        self.embedding_ = spectral_embedding(self.cluster_model_.affinity_matrix_, n_components=self.n_clusters,
                                             eigen_solver=self.cluster_model_.eigen_solver,
                                             random_state=self.random_state,
                                             eigen_tol=self.cluster_model_.eigen_tol, drop_first=False)
        return labels


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def _network_diffusion(self, Xs):
        S_dfs = []
        for X_idx, X in enumerate(Xs):
            dist_mat = dist2(X.values, X.values)
            S_mat = snf.compute.affinity_matrix(dist_mat, K=self.neighbor_size, mu=self.mu)
            S_df = pd.DataFrame(data=S_mat, index=self.original_order[X_idx], columns=self.original_order[X_idx])
            S_dfs.append(S_df)

        self.fused_networks_ = self._integrao_fuse(aff=S_dfs.copy(), dicts_common=self.dicts_common,
                                                  dicts_unique=self.dicts_unique, original_order=self.original_order,
                                                  neighbor_size=self.neighbor_size,
                                                  fusing_iteration=self.fusing_iteration)


    def _integrao_fuse(self, aff, dicts_common, dicts_unique, original_order, neighbor_size=20, fusing_iteration=20):
        newW = [0] * len(aff)
        for n, mat in enumerate(aff):
            # normalize affinity matrix based on strength of edges
            # mat = mat / np.nansum(mat, axis=1, keepdims=True)
            aff[n] = _stable_normalized_pd(mat)
            # aff[n] = check_symmetric(mat, raise_warning=False)

            # apply KNN threshold to normalized affinity matrix
            # We need to crop the intersecting samples from newW matrices
            neighbor_size = min(int(neighbor_size), mat.shape[0])
            newW[n] = _find_dominate_set(aff[n], neighbor_size)

        # If there is only one view, return it
        if len(aff) == 1:
            print("Only one view, return it directly")
            return newW

        for iteration in range(fusing_iteration):

            # Make a copy of the aff matrix for this iteration
            # goal is to update aff[n], but it is the average of all the defused matrices
            # Make a copy of add[n], and set it to 0
            aff_next = []
            for k in range(len(aff)):
                aff_temp = aff[k].copy()
                for col in aff_temp.columns:
                    aff_temp[col].values[:] = 0
                aff_next.append(aff_temp)

            for n, mat in enumerate(aff):
                # temporarily convert nans to 0 to avoid propagation errors
                nzW = newW[n]  # TODO: not sure this is a deep copy or not

                for j, mat_tofuse in enumerate(aff):
                    if n == j:
                        continue

                    # reorder mat_tofuse to have the common samples
                    mat_tofuse = mat_tofuse.reindex(
                        (sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)])),
                        axis=1,
                    )
                    mat_tofuse = mat_tofuse.reindex(
                        (sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)])),
                        axis=0,
                    )

                    # Next, let's crop mat_tofuse
                    num_common = len(dicts_common[(n, j)])
                    to_drop_mat = mat_tofuse.columns[
                                  num_common: mat_tofuse.shape[1]
                                  ].values.tolist()
                    mat_tofuse_crop = mat_tofuse.drop(to_drop_mat, axis=1)
                    mat_tofuse_crop = mat_tofuse_crop.drop(to_drop_mat, axis=0)

                    # Next, add the similarity from the view to fused to the current view identity matrix
                    nzW_identity = pd.DataFrame(
                        data=np.identity(nzW.shape[0]),
                        index=original_order[n],
                        columns=original_order[n],
                    )

                    mat_tofuse_union = nzW_identity + mat_tofuse_crop
                    mat_tofuse_union.fillna(0.0, inplace=True)
                    mat_tofuse_union = _scaling_normalized_pd(mat_tofuse_union,
                                                              ratio=mat_tofuse_crop.shape[0] / nzW_identity.shape[0])
                    mat_tofuse_union = check_symmetric(mat_tofuse_union, raise_warning=False)
                    mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=1)
                    mat_tofuse_union = mat_tofuse_union.reindex(original_order[n], axis=0)

                    # Now we are ready to do the diffusion
                    nzW_T = np.transpose(nzW)
                    aff0_temp = nzW.dot(
                        mat_tofuse_union.dot(nzW_T)
                    )  # Matmul is not working, but .dot() is good

                    #################################################
                    # Experimentally introduce a weighting machanisim, use the exponential weight; Already proved it's not a good idea
                    # num_com = mat_tofuse_crop.shape[0] / aff[n].shape[0]
                    # alpha = pow(2, num_com) - 1
                    # aff0_temp = alpha * aff0_temp + (1-alpha) * aff[n]

                    # aff0_temp = _B0_normalized(aff0_temp, alpha=normalization_factor)
                    aff0_temp = _stable_normalized_pd(aff0_temp)
                    # aff0_temp = check_symmetric(aff0_temp, raise_warning=False)

                    aff_next[n] = np.add(aff0_temp, aff_next[n])

                aff_next[n] = np.divide(aff_next[n], len(aff) - 1)
                # aff_next[n] = _stable_normalized_pd(aff_next[n])

            # put the value in aff_next back to aff
            for k in range(len(aff)):
                aff[k] = aff_next[k]

        for n, mat in enumerate(aff):
            aff[n] = _stable_normalized_pd(mat)
            # aff[n] = check_symmetric(mat, raise_warning=False)

        aff = [x.values for x in aff]
        return aff


    @staticmethod
    def _tsne_loss(P, activations):
        device = P.device
        n = activations.size(0)
        alpha = 1
        eps = 1e-12
        sum_act = torch.sum(torch.pow(activations, 2), 1)
        Q = (
                sum_act
                + sum_act.view([-1, 1])
                - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
        )
        Q = Q / alpha
        Q = torch.pow(1 + Q, -(alpha + 1) / 2)
        Q = Q * autograd.Variable(1 - torch.eye(n), requires_grad=False).to(device)
        Q = Q / torch.sum(Q)
        Q = torch.clamp(Q, min=eps)
        C = torch.log((P + eps) / (Q + eps))
        C = torch.sum(P * C)
        return C


class IntegrAOModule(nnModuleBase):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_dim = out_channels
        self.num_layers = num_layers

        num = len(in_channels)
        feature = []
        for i in range(num):
            model_sage = GraphSAGE(
                in_channels=self.in_channels[i],
                hidden_channels=self.hidden_channels,
                num_layers=self.num_layers,
                out_channels=self.output_dim,
                project=False)
            feature.append(model_sage)

        self.feature = nn.ModuleList(feature)

        self.feature_show = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.output_dim, self.output_dim),
        )


    def forward(self, Xs, average=True):
        z_all = []
        xs = Xs[0]
        edge_indices = Xs[1]
        idxs = Xs[2]
        ids = Xs[3].cpu().numpy()
        for X_idx, (X,edge_index,idx) in enumerate(zip(xs, edge_indices, idxs)):
            X = pd.DataFrame(X.cpu().numpy(), index=ids).loc[idx[0].cpu().numpy()]
            X = torch.from_numpy(X.values).type(torch.float32)
            z = self.feature[X_idx](X, edge_index[0])
            z = self.feature_show(z)
            z_all.append(z)
        if average:
            mean_z = np.zeros((len(np.unique(np.concatenate([idx[0] for idx in idxs]))), z_all[0].shape[1]))
            mean_z = pd.DataFrame(mean_z)
            ones_z = mean_z.copy()
            for X_idx, (X, idx, z) in enumerate(zip(xs, idxs, z_all)):
                idx = idx[0].numpy()
                mean_z.loc[idx] += z.numpy()
                ones_z.loc[idx] += 1
            mean_z /= ones_z
            z_all = torch.from_numpy(mean_z.values).type(torch.float32)

        return z_all