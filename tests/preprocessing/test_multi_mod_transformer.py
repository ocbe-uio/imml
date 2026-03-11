import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from imml.preprocessing import MMTransformer

@pytest.fixture
def sample_data():
    X1 = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4],
        'feature2': [np.nan, 2, 3, 4]
    }, index=['a', 'b', 'c', 'd'])

    X2 = pd.DataFrame({
        'feature1': [5, np.nan, 7, 8],
        'feature2': [1, 2, np.nan, 4]
    }, index=['a', 'b', 'c', 'd'])

    return [X1, X2]

def test_single_transformer(sample_data):
    transformer = MMTransformer(transformer=SimpleImputer(strategy='mean'))
    transformer.fit(sample_data)
    transformed_Xs = transformer.transform(sample_data)

    assert len(transformed_Xs) == 2
    for transformer_,X_transformed in zip(transformer.transformer_list_, transformed_Xs):
        assert np.all(~np.isnan(X_transformed))
        try:
            check_is_fitted(transformer_)
        except:
            raise

def test_multiple_transformers(sample_data):
    transformers = [SimpleImputer(strategy='mean'), StandardScaler()]
    transformer = MMTransformer(transformer=transformers)
    transformer.fit(sample_data)
    transformed_Xs = transformer.transform(sample_data)

    assert len(transformed_Xs) == 2
    for transformer_ in transformer.transformer_list_:
        try:
            check_is_fitted(transformer_)
        except:
            raise

def test_single_transformer_different_mods(sample_data):
    transformer = MMTransformer(transformer=SimpleImputer(strategy='mean'))
    transformer.fit(sample_data)
    transformed_Xs = transformer.transform(sample_data)

    assert all(isinstance(X, np.ndarray) for X in transformed_Xs)
    assert all(np.all(~np.isnan(X)) for X in transformed_Xs)

def test_invalid_transformer():
    with pytest.raises(ValueError, match="transformer must be a scikit-learn transformer like object"):
        MMTransformer(transformer="a")
    with pytest.raises(ValueError, match="transformer must be a scikit-learn transformer like object"):
        MMTransformer(transformer=["a"])

def test_fit_no_transform(sample_data):
    transformer = MMTransformer(transformer=SimpleImputer(strategy='mean'))
    transformer.fit(sample_data)
    assert len(transformer.transformer_list_) == len(sample_data)
