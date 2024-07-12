import pandas as pd
import torch
import torch.nn
from sklearn.decomposition import FastICA
import numpy as np
import lightning as pl

from ._deepmf.sparselinear import _SparseLinear


class DeepMFDataset(torch.utils.data.Dataset):

    def __init__(self, X, transform = None, target_transform = None):
        self.X = X
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        X = self.X
        label = X[idx, :]
        Xi = torch.tensor([idx], dtype=torch.long).unsqueeze(0)
        Xv = torch.ones(1, dtype=torch.float)
        X = torch.sparse_coo_tensor(Xi, Xv, (len(X),))
        label = label.to(X.dtype)

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            label = self.target_transform(label)
        return X, label


class DeepMF(pl.LightningModule):
    r"""
    DeepMF, a deep neural network-based factorization model, elucidates the association between
    feature-associated and sample-associated latent matrices, and is robust to noisy and missing values. It only accepts
    a single view as input, so multi-view datasets should be concatenated before.

    It should be used with PyTorch Lightning. The class DeepMFDataset provides a Dataset class for this algorithm. X has
    dimensions MxN, where M is the number of features and N is the number of samples.

    Parameters
    ----------
    latent_dim : int, default=10
        Number of dimensions to keep.
    n_layers : int, default=3
        Number of layers in the deep encoder.
    learning_rate : float, default=1e-2
        Learning rate.
    alpha: float, default=0.01
        Hyperparameter to control the loss function.
    neighbor_proximity : str, default='Lap'
        Penalty when similar features and similar samples are embedded far away in the latent space. One of 'Lap', 'MSE'
         or 'KL'.
    loss_fun : func, default=torch.nn.MSELoss()
        Loss function.
    sigmoid : bool, default=False
        If applying sigmoid function to the last layer.

    Attributes
    ----------
    model_ : torch.nn.Sequential
        Torch model.
    U_ : torch.tensor
        Feature latent factor matrix.
    V_: torch.tensor
        Sample latent factor matrix.

    References
    ----------
    [paper] Chen, L., Xu, J. & Li, S.C. DeepMF: deciphering the latent patterns in omics profiles with a deep learning
            method. BMC Bioinformatics 20 (Suppl 23), 648 (2019). https://doi.org/10.1186/s12859-019-3291-6.
    [code] https://github.com/paprikachan/DeepMF

    Examples
    --------
    #todo
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.decomposition import DeepMF, DeepMFDataset
    >>> from imvc.preprocessing import MultiViewTransformer, ConcatenateViews
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler, FunctionTransformer
    >>> from sklearn.cluster import KMeans
    >>> from torch.utils.data import DataLoader
    >>> from pytorch_lightning import Trainer

    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> pipeline = make_pipeline(ConcatenateViews(), StandardScaler(), FunctionTransformer(lambda x: torch.from_numpy(x).float().cuda().t()))
    >>> transformed_X = pipeline.fit_transform(Xs)
    >>> train_data = DeepMFDataset(X= transformed_X)
    >>> train_dataloader = DataLoader(dataset= train_data, batch_size= 50, shuffle=True)
    >>> trainer = Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
    >>> model = DeepMF(X=transformed_X)
    >>> trainer.fit(model, train_dataloader)
    >>> train_dataloader = DataLoader(dataset= train_data, batch_size= 50, shuffle=False)
    >>> transformed_X = model.transform(transformed_X)
    >>> pipeline = make_pipeline(FunctionTransformer(lambda x: x.cpu().detach().numpy()), StandardScaler(), KMeans(n_clusters=3))
    >>> labels = pipeline.fit_predict(transformed_X)
    """

    def __init__(self, X = None, latent_dim: int =10, n_layers: int = 3, learning_rate: float = 1e-2, alpha: float = 0.01,
                 neighbor_proximity='Lap', loss_fun = torch.nn.MSELoss(), sigmoid: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.neighbor_proximity = neighbor_proximity
        self.loss_fun = loss_fun
        self.sigmoid = sigmoid
        self.transform_ = None
        if X is not None:
            self._init_model(X)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=0.0001)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model_(x)
        y_pred, local_y = self._clean_y(y_pred, y)
        loss = self.loss_fun(y_pred, local_y)
        u_loss, v_loss = self._U_V_loss(self.Su, self.Sv, self.Du, self.Dv)
        loss = (1 - self.alpha) * loss + self.alpha * (u_loss + v_loss)
        self.U_, self.V_ = self._load_U_V()
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model_(x)
        y_pred, local_y = self._clean_y(y_pred, y)
        loss = self.loss_fun(y_pred, local_y)
        u_loss, v_loss = self._U_V_loss(self.Su, self.Sv, self.Du, self.Dv)
        loss = (1 - self.alpha) * loss + self.alpha * (u_loss + v_loss)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model_(x)
        y_pred, local_y = self._clean_y(y_pred, y)
        loss = self.loss_fun(y_pred, local_y)
        u_loss, v_loss = self._U_V_loss(self.Su, self.Sv, self.Du, self.Dv)
        loss = (1 - self.alpha) * loss + self.alpha * (u_loss + v_loss)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        pred = self.model_(x)
        return pred


    def _get_similarity(self, data, row_or_col, n=15):
        self.data_isnan = torch.isnan(data)
        if torch.any(self.data_isnan):
            D = self._get_distance_with_nan(data, self.data_isnan, row_or_col)
        else:
            D = self._get_distance(data, row_or_col)
        S = 1 / (1 + D)
        N, _ = D.shape
        if not n:
            n = int(N / self.K)
        cutoff = D[range(N), torch.argsort(D, dim=1)[:, n-1]]
        Indicator = torch.zeros(N, N, dtype=torch.float).to(S.device)
        for i in range(N):
            for j in range(N):
                if D[i, j] <= cutoff[i] or D[i, j] <= cutoff[j]:
                    Indicator[i, j] = 1
        # sigma = D.std()/2
        # S = torch.exp(-D/(2*sigma*sigma))
        S = Indicator * S
        return S, D


    def _get_distance(self, data, row_or_col):
        M, N = data.shape
        if row_or_col == 'row':
            G = torch.mm(data, data.t())
        else:
            G = torch.mm(data.t(), data)
        g = torch.diag(G)
        if row_or_col == 'row':
            x = g.repeat(M, 1)
        else:
            x = g.repeat(N, 1)
        D = x + x.t() - 2*G
        return D


    def _get_distance_with_nan(self, o_data, data_isnan, row_or_col):
        data = o_data.clone().detach()
        data[torch.isnan(data)] = 0
        D = self._get_distance(data, row_or_col)
        M, N = data.shape
        if row_or_col == 'row':
            row_or_col_nan = torch.sum(data_isnan, dim=1).repeat(M, 1)
            penalty = torch.max(data) * 1.0 / N
        else:
            row_or_col_nan = torch.sum(data_isnan, dim=0).repeat(N, 1)
            penalty = torch.max(data) * 1.0 / M
        if penalty == 0:
            penalty = 1
        nan_penalty = penalty * (row_or_col_nan + row_or_col_nan.t())
        nan_penalty = nan_penalty.float()
        D = D + nan_penalty - torch.diag(torch.diag(nan_penalty))

        return D


    def _U_V_loss(self, Su, Sv, Du, Dv):
        U, V = self._load_U_V()

        if self.neighbor_proximity == 'Lap':
            u_loss = self._lap_loss(U, Su, 'row')
            v_loss = self._lap_loss(V, Sv, 'col')
        elif self.neighbor_proximity == 'MSE':
            u_loss = torch.sum(torch.pow(self._get_distance(U, 'row') - Du, 2)) / (self.M * self.M) * 0.01
            v_loss = torch.sum(torch.pow(self._get_distance(V, 'col') - Dv, 2)) / (self.N * self.N) * 0.01
        elif self.neighbor_proximity == 'KL':
            u_loss = self._kl_loss(1 / (1+Du), 1 / (1 + self._get_distance(U, 'row')))
            v_loss = self._kl_loss(1 / (1+Dv), 1 / (1 + self._get_distance(V, 'col')))
        else:
            u_loss = 0
            v_loss = 0

        return u_loss, v_loss

    def _kl_loss(self, P, Q):
        # return torch.sum(P*torch.log2(P/Q))

        return torch.log(torch.sum((P*P/Q)-P+Q))

    def _lap_loss(self, W, Sw, row_or_col):
        return torch.sum(self._get_distance(W, row_or_col) * Sw)

    def _initial_U_V(self, data):
        model = FastICA(n_components=self.K)

        U = model.fit_transform(np.nan_to_num(data))
        U = torch.tensor(U, dtype=torch.float)

        V = model.fit_transform(np.nan_to_num(data).T)
        V = torch.tensor(V, dtype=torch.float)

        self.model_[0].weight.data = U.t()
        self.model_[-1].weight.data = V


    def _load_U_V(self):
        U = self.model_[0].weight.t()
        V = self.model_[-1].weight.t()
        return U, V


    def _clean_y(self, y_pred, y):
        y_pred[torch.isnan(y)] = 0
        y[torch.isnan(y)] = 0
        return y_pred, y


    def transform(self, X):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        X : array-likes of shape (n_samples, n_features)
                New data to transform.

        Returns
        -------
        transformed_X : array-like of shape (n_samples, latent_dim)
            The projected data.
        """
        transformed_X = torch.mm(X.T, self.U_)
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X.numpy(), index= X.index)
        return transformed_X


    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        if "X" in parameters.keys():
            self._init_model(parameters["X"])
        return self


    def _init_model(self, X):
        self.M, self.N = X.shape
        layers = [_SparseLinear(self.M, self.latent_dim)]
        for i in range(self.n_layers):
            layers.append(torch.nn.Linear(self.latent_dim, self.latent_dim))
            if self.n_layers > 1:
                layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.latent_dim, self.N))
        if self.sigmoid:
            layers.append(torch.nn.Sigmoid())

        self.model_ = torch.nn.Sequential(*layers)
        self.Su, self.Du = self._get_similarity(X, 'row')
        self.Sv, self.Dv = self._get_similarity(X, 'col')
