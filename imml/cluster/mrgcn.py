# License: BSD-3-Clause

from sklearn.cluster import KMeans


try:
    import torch
    from torch import nn
    import lightning as L
    from torch.nn import functional as F
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

DLBaseeModule = L.LightningModule if deepmodule_installed else object

class MRGCN(DLBaseeModule):
    r"""
    Multi-Reconstruction Graph Convolutional Network (MRGCN). [#mrgcnpaper]_ [#mrgcncode]_

    MRGCN encodes and reconstructs data and similarity relationships from multiple sources simultaneously,
    consolidating them into a shared latent embedding space. Additionally, MRGCN utilizes an indicator matrix to
    represent the presence of missing modalities, effectively merging the processing of complete and incomplete multi-modal
    data within a single unified framework.

    Incomplete samples should be filled with 0.

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    Xs : list of array-likes objects, default=None
        Multi-modal dataset. It will be used to create the neural network architecture.
    k_num : int, default=10
        Number of neighbors to use.
    learning_rate : float, default=1e-3
        Learning rate.
    reg2 : float, default=1.
        Trade-off parameter to control the graph structure reconstruction.
    reg3 : float, default=1.
        Trade-off parameter to control the self-supervised learning mechanism.

    Attributes
    ----------
    kmeans_ : KMeans object
        Scikit-learn KMeans object.

    References
    ----------
    .. [#mrgcnpaper] Bo Yang, Yan Yang, Meng Wang, Xueping Su, MRGCN: cancer subtyping with multi-reconstruction
                    graph convolutional network using full and partial multi-omics dataset, Bioinformatics, Volume 39,
                    Issue 6, June 2023, btad353, https://doi.org/10.1093/bioinformatics/btad353
    .. [#mrgcncode] https://github.com/Polytech-bioinf/MRGCN

    Example
    --------
    >>> import numpy as np
    >>> import torch
    >>> from imml.cluster import MRGCN
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import MRGCNDataset
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> train_data = MRGCNDataset(Xs=Xs)
    >>> train_dataloader = DataLoader(dataset=train_data)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = MRGCN(Xs=Xs, n_clusters=2)
    >>> trainer.fit(estimator, train_dataloader)
    >>> labels = trainer.predict(estimator, train_dataloader)[0]
    """

    def __init__(self, n_clusters: int = 8, Xs = None, k_num:int = 10, learning_rate:float = 0.001, reg2:float = 1.,
                 reg3:float = 1.):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)
        super().__init__()

        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list of array-likes objects. A {type(Xs)} was passed.")
        if not isinstance(k_num, int):
            raise ValueError(f"Invalid k_num. It must be an int. A {type(k_num)} was passed.")
        if k_num < 1:
            raise ValueError(f"Invalid k_num. It must be a positive number. {k_num} was passed.")
        if not isinstance(learning_rate, float):
            raise ValueError(f"Invalid learning_rate. It must be a float. A {type(learning_rate)} was passed.")
        if learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate. It must be a positive number. {learning_rate} was passed.")
        if not isinstance(reg2, float):
            raise ValueError(f"Invalid reg2. It must be a float. A {type(reg2)} was passed.")
        if not isinstance(reg3, float):
            raise ValueError(f"Invalid reg3. It must be a float. A {type(reg3)} was passed.")

        self.data = Xs
        self.n_features_ = [X.shape[1] for X in Xs]
        self.n_mods_ = len(Xs)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.n_clusters = n_clusters
        we = []
        self.kmeans_ = KMeans(n_clusters=n_clusters, n_init="auto")
        self.reg2 = reg2
        self.reg3 = reg3
        self.gs = []
        self.ss = []

        for idx, X in enumerate(Xs):
            n_features_i = X.shape[1]
            g = self._get_kNNgraph2(X, k_num)
            self.gs.append(g)
            s = self._comp(g)
            self.ss.append(s)
            ind = torch.any(X, 1).int()
            we.append(ind)

            dims = []
            linshidim = round(n_features_i * 0.8)
            linshidim = int(linshidim)
            dims.append(linshidim)
            linshidim = round(min(self.n_features_) * 0.8)
            linshidim = int(linshidim)
            dims.append(linshidim)

            enc_1 = nn.Linear(n_features_i, dims[0])
            enc_2 = nn.Linear(dims[0], dims[1])
            dec_1 = nn.Linear(dims[1], dims[0])
            dec_2 = nn.Linear(dims[0], n_features_i)
            weight1 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims[1], dims[1])))
            weight_1 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(dims[1], dims[1])))

            setattr(self, f"enc{idx}_1", enc_1)
            setattr(self, f"enc{idx}_2", enc_2)
            setattr(self, f"dec{idx}_1", dec_1)
            setattr(self, f"dec{idx}_2", dec_2)
            setattr(self, f"weight{idx}", weight1)
            setattr(self, f"weight_{idx}", weight_1)

        self.we = torch.stack(we).float()


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        z = self._embedding(batch=batch)
        loss_x = 0
        loss_a = 0
        for X_idx in range(self.n_mods_):
            weight = getattr(self, f"weight{X_idx}")
            a = torch.sigmoid(torch.matmul(torch.matmul(z, weight), z.T))
            loss_a += self.criterion(a, self.gs[X_idx].to(a.device))
            weight_ = getattr(self, f"weight_{X_idx}")
            h = torch.tanh(torch.matmul(z, weight_))
            dec_1 = getattr(self, f"dec{X_idx}_1")
            h_1 = torch.tanh(dec_1(torch.matmul(self.ss[X_idx].to(h), h)))
            dec_2 = getattr(self, f"dec{X_idx}_2")
            h_2 = torch.tanh(dec_2(torch.matmul(self.ss[X_idx].to(h_1), h_1)))
            loss_x += self.criterion(h_2, batch[X_idx])

        self.kmeans_.fit(z.detach().cpu().numpy())
        cluster_layer = torch.tensor(self.kmeans_.cluster_centers_).to(z.device)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2))
        q = q.pow(1)
        q = torch.t(torch.t(q) / torch.sum(q, 1))
        weight = q ** 2 / q.sum(0)
        p = torch.t(torch.t(weight) / weight.sum(1))
        loss_kl = F.kl_div(q.log(), p, reduction='batchmean')

        loss = loss_x + self.reg2 * loss_a + self.reg3 * loss_kl
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return self.training_step(batch, batch_idx=None)


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        return self.training_step(batch, batch_idx=None)


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        z = self._embedding(batch=batch)
        z = z.detach().cpu().numpy()
        pred = self.kmeans_.predict(z)
        return pred


    def on_fit_end(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        z = self._embedding(batch=self.data)
        z = z.detach().cpu().numpy()
        self.kmeans_.fit(z)
        del self.data


    @staticmethod
    def _get_kNNgraph2(data, K_num):
        # each row of data is a sample

        x_norm = torch.reshape(torch.sum(torch.square(data), 1), [-1, 1])  # column vector
        x_norm2 = torch.reshape(torch.sum(torch.square(data), 1), [1, -1])  # column vector
        dists = x_norm - 2 * torch.matmul(data, torch.transpose(data, 0, 1)) + x_norm2
        num_sample = data.shape[0]
        graph = torch.zeros((num_sample, num_sample))
        for i in range(num_sample):
            distance = dists[i, :]
            small_index = torch.argsort(distance)
            graph[i, small_index[:K_num]] = 1
        graph = graph - torch.diag(torch.diag(graph))
        resultgraph = torch.maximum(graph, torch.transpose(graph, 0, 1))
        return resultgraph


    @staticmethod
    def _comp(g):
        g = g + torch.eye(g.shape[0])
        g = torch.tensor(g)
        d = torch.diag(g.sum(dim=1))
        d = torch.tensor(d)
        s = pow(d, -0.5)
        where_are_inf = torch.isinf(s)
        s[where_are_inf] = 0
        s = torch.matmul(torch.matmul(s, g), s).to(torch.float32)
        return s


    def _embedding(self, batch):
        summ = 0
        for X_idx, X in enumerate(batch):
            enc_1 = getattr(self, f"enc{X_idx}_1")
            enc_2 = getattr(self, f"enc{X_idx}_2")
            output_1 = torch.tanh(enc_1(torch.matmul(self.ss[X_idx].to(X.device), X)))
            output_2 = torch.tanh(enc_2(torch.matmul(self.ss[X_idx].to(output_1.device), output_1)))
            summ += torch.diag(self.we[X_idx, :].to(output_2.device)).mm(output_2)

        wei = 1 / torch.sum(self.we, 0)
        z = torch.diag(wei.to(summ.device)).mm(summ)
        return z
