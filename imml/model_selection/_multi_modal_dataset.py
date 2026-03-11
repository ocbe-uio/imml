# License: BSD-3-Clause

class _MultiModalDataset:

    def __init__(self, Xs):
        self.Xs = list(Xs)
        self.n_samples = len(self.Xs[0])
        self._take = self.take_iloc if hasattr(self.Xs[0], "iloc") else self.take_loc


    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):
        return _MultiModalDataset([self._take(X=X, idx=idx) for X in self.Xs])


    def take_iloc(self, X, idx):
        return X.iloc[idx]


    def take_loc(self, X, idx):
        return X[idx]


    def to_list(self):
        return self.Xs
