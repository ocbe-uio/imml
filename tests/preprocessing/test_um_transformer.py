import pytest
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.utils.validation import check_is_fitted

from imml.preprocessing import UMTransformer
from imml import deepmodule_installed

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        return Xs_pandas, Xs_numpy, Xs_torch
    return Xs_pandas, Xs_numpy


def test_transformer(sample_data):
    transformer = UMTransformer(transformer=KNNImputer(n_neighbors=1))
    for Xs in sample_data:
        transformer.fit(Xs)
        transformed_Xs = transformer.transform(Xs)

        assert len(transformed_Xs) == 2
        check_is_fitted(transformer.transformer)
        for X, transformed_X in zip(Xs, transformed_Xs):
            assert np.equal(X.shape, transformed_X.shape).all()
            assert np.all(~np.isnan(transformed_X))


def test_invalid_transformer():
    with pytest.raises(ValueError, match="transformer must be a scikit-learn transformer like object"):
        UMTransformer(transformer="a")


def test_example():
    import numpy as np
    import pandas as pd
    from imml.preprocessing import UMTransformer
    from sklearn.impute import KNNImputer
    Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    transformer = UMTransformer(transformer = KNNImputer())
    transformer.fit_transform(Xs)
