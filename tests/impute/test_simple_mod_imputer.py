import pytest
import numpy as np
import pandas as pd
from imml.impute import SimpleModImputer, simple_mod_imputer

@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    indxs = [[2, 4], 1]
    means = [[0.65893904, 0.58486764, 0.56949316], [0.81022048, 0.48690986]]
    return Xs_pandas, Xs_numpy, indxs, means

def test_mean_imputation(sample_data):
    indxs, means = sample_data[2:]
    for Xs in sample_data[:2]:
        imputer = SimpleModImputer(value='mean')
        transformed_Xs = imputer.fit_transform(Xs)
        assert len(transformed_Xs) == 2
        assert len(transformed_Xs) == len(imputer.features_mod_mean_list_)
        for X_transformed,idx, features_mod_mean, mes in zip(transformed_Xs, indxs,
                                                              imputer.features_mod_mean_list_, means):
            assert X_transformed.shape[1] == len(features_mod_mean)
            assert not np.isnan(X_transformed).any().any()
            assert np.allclose(features_mod_mean, mes)
            if isinstance(X_transformed, pd.DataFrame):
                assert np.allclose(X_transformed.iloc[idx], mes)
            elif isinstance(X_transformed, np.ndarray):
                assert np.allclose(X_transformed[idx], mes)

def test_zeros_imputation(sample_data):
    indxs, means = sample_data[2:]
    for Xs in sample_data[:2]:
        imputer = SimpleModImputer(value='zeros')
        transformed_Xs = imputer.fit_transform(Xs)
        assert len(transformed_Xs) == 2
        for X_transformed,idx in zip(transformed_Xs, indxs):
            if isinstance(X_transformed, pd.DataFrame):
                assert (X_transformed.iloc[idx] == 0).all().all()
            elif isinstance(X_transformed, np.ndarray):
                assert (X_transformed[idx] == 0).all().all()

def test_invalid_value():
    with pytest.raises(ValueError, match="Invalid value. Expected one of:"):
        SimpleModImputer(value='invalid')

def test_transform_without_fit(sample_data):
    imputer = SimpleModImputer(value='mean')
    with pytest.raises(AttributeError):
        imputer.transform(sample_data[0])

def test_mean_imputation_function(sample_data):
    indxs, means = sample_data[2:]
    for Xs in sample_data[:2]:
        transformed_Xs = simple_mod_imputer(Xs, value='mean')
        assert len(transformed_Xs) == 2
        for X_transformed,idx, mes in zip(transformed_Xs, indxs, means):
            assert not np.isnan(X_transformed).any().any()
            if isinstance(X_transformed, pd.DataFrame):
                assert np.allclose(X_transformed.iloc[idx], mes)
            elif isinstance(X_transformed, np.ndarray):
                assert np.allclose(X_transformed[idx], mes)

def test_zeros_imputation_function(sample_data):
    indxs, means = sample_data[2:]
    for Xs in sample_data[:2]:
        transformed_Xs = simple_mod_imputer(Xs, value='zeros')
        assert len(transformed_Xs) == 2
        for X_transformed,idx in zip(transformed_Xs, indxs):
            assert not np.isnan(X_transformed).any().any()
            if isinstance(X_transformed, pd.DataFrame):
                assert (X_transformed.iloc[idx] == 0).all().all()
            elif isinstance(X_transformed, np.ndarray):
                assert (X_transformed[idx] == 0).all().all()

def test_invalid_value_function(sample_data):
    Xs_pandas, Xs_numpy, indxs, means = sample_data
    with pytest.raises(ValueError, match="Invalid value. Expected one of:"):
        simple_mod_imputer(Xs_pandas, value='invalid')


if __name__ == "__main__":
    pytest.main()