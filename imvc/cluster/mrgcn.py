import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl
from torch.nn import functional as F


class MRGCNDataset(torch.utils.data.Dataset):

    def __init__(self, Xs, transform = None):
        self.Xs = Xs
        self.transform = transform


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        if self.transform is not None:
            Xs = [self.transform[X_idx](X[idx]) for X_idx,X in enumerate(self.Xs)]
        else:
            Xs = [X[idx] for X in self.Xs]
        Xs = tuple(Xs)
        return Xs


class MRGCN(pl.LightningModule):
    def __init__(self, kmeans, Xs = None, k_num:int = 10, learning_rate:float = 0.001, reg2:int = 1, reg3:int = 1, **args):
        super(MRGCN, self).__init__()
        self.data = Xs
        self.n_features_ = [X.shape[1] for X in Xs]
        self.n_views_ = len(Xs)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss(reduction='sum')
        we = []
        self.kmeans = kmeans
        self.reg2 = reg2
        self.reg3 = reg3
        self.gs = []
        self.ss = []

        for idx, X in enumerate(Xs):
            n_features_i = X.shape[1]
            g = self.get_kNNgraph2(X, k_num)
            self.gs.append(g)
            s = self.comp(g)
            self.ss.append(s)
            ind = torch.any(X, axis=1).int()
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
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)


    def training_step(self, batch, batch_idx):
        z = self.embedding(batch=batch)
        loss_x = 0
        loss_a = 0
        for view_idx in range(self.n_views_):
            weight = getattr(self, f"weight{view_idx}")
            a = torch.sigmoid(torch.matmul(torch.matmul(z, weight), z.T))
            loss_x += self.criterion(a, self.gs[view_idx])
            weight_ = getattr(self, f"weight_{view_idx}")
            h = torch.tanh(torch.matmul(z, weight_))
            dec_1 = getattr(self, f"dec{view_idx}_1")
            h_1 = torch.tanh(dec_1(torch.matmul(self.ss[view_idx], h)))
            dec_2 = getattr(self, f"dec{view_idx}_2")
            h_2 = torch.tanh(dec_2(torch.matmul(self.ss[view_idx], h_1)))
            loss_x += self.criterion(h_2, batch[view_idx])

        self.kmeans.fit(z.detach().numpy())
        cluster_layer = torch.tensor(self.kmeans.cluster_centers_).to(z.device)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2))
        q = q.pow(1)
        q = torch.t(torch.t(q) / torch.sum(q, 1))
        weight = q ** 2 / q.sum(0)
        p = torch.t(torch.t(weight) / weight.sum(1))
        loss_kl = F.kl_div(q.log(), p, reduction='batchmean')

        loss = loss_x + self.reg2 * loss_a + self.reg3 * loss_kl
        return loss


    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)


    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        z = self.embedding(batch=batch)
        z = z.detach().numpy()
        pred = self.kmeans.predict(z)
        return pred


    def on_fit_end(self):
        z = self.embedding(batch=self.data)
        z = z.detach().numpy()
        self.kmeans.fit(z)


    @staticmethod
    def get_kNNgraph2(data, K_num):
        # each row of data is a sample

        x_norm = np.reshape(torch.sum(np.square(data), 1), [-1, 1])  # column vector
        x_norm2 = np.reshape(torch.sum(np.square(data), 1), [1, -1])  # column vector
        dists = x_norm - 2 * np.matmul(data, np.transpose(data)) + x_norm2
        num_sample = data.shape[0]
        graph = torch.zeros((num_sample, num_sample))
        for i in range(num_sample):
            distance = dists[i, :]
            small_index = np.argsort(distance)
            graph[i, small_index[0:K_num]] = 1
        graph = graph - np.diag(np.diag(graph))
        resultgraph = np.maximum(graph, np.transpose(graph))
        return resultgraph


    @staticmethod
    def comp(g):
        g = g + np.identity(g.shape[0])
        g = torch.tensor(g)
        d = np.diag(g.sum(axis=1))
        d = torch.tensor(d)
        s = pow(d, -0.5)
        where_are_inf = np.isinf(s)
        s[where_are_inf] = 0
        s = torch.matmul(torch.matmul(s, g), s).to(torch.float32)
        return s


    def embedding(self, batch):
        summ = 0
        for view_idx, view_data in enumerate(batch):
            enc_1 = getattr(self, f"enc{view_idx}_1")
            enc_2 = getattr(self, f"enc{view_idx}_2")
            output_1 = torch.tanh(enc_1(torch.matmul(self.ss[view_idx], view_data)))
            output_2 = torch.tanh(enc_2(torch.matmul(self.ss[view_idx], output_1)))
            summ += torch.diag(self.we[view_idx, :]).mm(output_2)

        wei = 1 / torch.sum(self.we, 0)
        z = torch.diag(wei).mm(summ).detach().numpy()
        return z
