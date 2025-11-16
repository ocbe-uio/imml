import pytest
import numpy as np
import pandas as pd

from imml.explore import get_n_mods, get_n_samples_by_mod, get_com_samples, \
    get_incom_samples, get_samples, get_samples_by_mod, get_missing_samples_by_mod, \
    get_n_com_samples, get_n_incom_samples, get_pct_com_samples, get_pct_incom_samples, \
    get_summary


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    observed_mod_indicator = pd.DataFrame({
        0: [True, True, False, True, False],
        1: [True, False, True, True, True]
    })
    observed_mod_indicator = observed_mod_indicator.values
    return Xs_pandas, Xs_numpy, observed_mod_indicator


def test_get_summary(sample_data):
    for Xs in sample_data[:2]:
        # one_row=True with compute_pct=True
        summary = get_summary(Xs=Xs, one_row=True, compute_pct=True)
        assert isinstance(summary, dict)
        assert len(summary.keys()) == 6
        assert isinstance(summary["Complete samples"], int)
        assert isinstance(summary["Incomplete samples"], int)
        assert summary["Complete samples"] == 2
        assert summary["Incomplete samples"] == 3
        assert summary["Complete samples"] + summary["Incomplete samples"] == len(Xs[0])
        assert len(summary["Observed samples per modality"]) == len(Xs)
        assert len(summary["Missing samples per modality"]) == len(Xs)
        assert summary["Observed samples per modality"] == [3, 4]
        assert summary["Missing samples per modality"] == [2, 1]
        assert len(summary["% Observed samples per modality"]) == len(Xs)
        assert any([i <= 100 for i in summary["% Observed samples per modality"]])
        assert summary["% Observed samples per modality"] == [60, 80]
        assert len(summary["% Missing samples per modality"]) == len(Xs)
        assert any([i <= 100 for i in summary["% Missing samples per modality"]])
        assert summary["% Missing samples per modality"] == [40, 20]

        # one_row=True with compute_pct=False (branch coverage)
        summary_npct = get_summary(Xs=Xs, one_row=True, compute_pct=False)
        assert isinstance(summary_npct, dict)
        assert len(summary_npct.keys()) == 6
        assert summary_npct["Complete samples"] == 2
        assert summary_npct["Incomplete samples"] == 3
        assert summary_npct["Observed samples per modality"] == [3, 4]
        assert summary_npct["Missing samples per modality"] == [2, 1]
        assert summary_npct["% Observed samples per modality"] == [60, 80]
        assert summary_npct["% Missing samples per modality"] == [40, 20]

        # one_row=False with modalities=None and compute_pct=False
        summary_by_mod = get_summary(Xs=Xs, one_row=False, compute_pct=False)
        assert set(summary_by_mod.keys()) == set([0, 1, "Total"])  # two mods + total
        # Per-modality counts
        assert summary_by_mod[0]["Complete samples"] == 3
        assert summary_by_mod[0]["Missing samples"] == 2
        assert summary_by_mod[0]["Incomplete samples"] == 2
        assert summary_by_mod[1]["Complete samples"] == 4
        assert summary_by_mod[1]["Missing samples"] == 1
        assert summary_by_mod[1]["Incomplete samples"] == 1
        # Totals
        assert summary_by_mod["Total"]["Complete samples"] == 2
        assert summary_by_mod["Total"]["Missing samples"] == 3
        assert summary_by_mod["Total"]["Incomplete samples"] == 3

        # one_row=False with provided modality names and compute_pct=True
        summary_pct = get_summary(Xs=Xs, mod_names=["mod1", "mod2"], one_row=False, compute_pct=True)
        assert set(summary_pct.keys()) == set(["mod1", "mod2", "Total"])  # named mods + total
        # Check percentages added correctly (counts/5*100)
        assert summary_pct["mod1"]["% Complete samples"] == (3 / 5) * 100
        assert summary_pct["mod1"]["% Missing samples"] == (2 / 5) * 100
        assert summary_pct["mod1"]["% Incomplete samples"] == (2 / 5) * 100
        assert summary_pct["mod2"]["% Complete samples"] == (4 / 5) * 100
        assert summary_pct["mod2"]["% Missing samples"] == (1 / 5) * 100
        assert summary_pct["mod2"]["% Incomplete samples"] == (1 / 5) * 100
        assert summary_pct["Total"]["% Complete samples"] == (2 / 5) * 100
        assert summary_pct["Total"]["% Missing samples"] == (3 / 5) * 100
        assert summary_pct["Total"]["% Incomplete samples"] == (3 / 5) * 100


def test_get_n_mods(sample_data):
    for Xs in sample_data[:2]:
        n_mods = get_n_mods(Xs)
        assert n_mods == len(Xs)


def test_get_n_samples_by_mod(sample_data):
    for Xs in sample_data[:2]:
        n_samples_by_mod = get_n_samples_by_mod(Xs)
        values_to_compare = [3,4]
        assert all(n_samples_by_mod == values_to_compare)


def test_get_com_samples(sample_data):
    for Xs in sample_data[:2]:
        complete_samples = get_com_samples(Xs)
        assert len(complete_samples) == 2
        values_to_compare = pd.Index([0,3])
        assert complete_samples.equals(values_to_compare)


def test_get_incom_samples(sample_data):
    for Xs in sample_data[:2]:
        incomplete_samples = get_incom_samples(Xs)
        assert len(incomplete_samples) == 3
        values_to_compare = pd.Index([1,2,4])
        assert incomplete_samples.equals(values_to_compare)


def test_get_samples(sample_data):
    for Xs in sample_data[:2]:
        sample_names = get_samples(Xs)
        assert len(sample_names) == 5
        values_to_compare = pd.Index(range(len(sample_names)))
        assert sample_names.equals(values_to_compare)


def test_get_samples_by_mod(sample_data):
    for Xs in sample_data[:2]:
        samples_by_mod = get_samples_by_mod(Xs, return_as_list=True)
        assert len(samples_by_mod) == len(Xs)
        values_to_compare = [[0, 1, 3], [0, 2, 3, 4]]
        for i in range(len(samples_by_mod)):
            assert samples_by_mod[i].equals(pd.Index(values_to_compare[i]))
        samples_by_mod = get_samples_by_mod(Xs, return_as_list=False)
        assert len(samples_by_mod) == len(Xs)
        for i in range(len(samples_by_mod)):
            assert samples_by_mod[i].equals(pd.Index(values_to_compare[i]))


def test_get_missing_samples_by_mod(sample_data):
    for Xs in sample_data[:2]:
        missing_samples_by_mod = get_missing_samples_by_mod(Xs, return_as_list=True)
        assert len(missing_samples_by_mod) == len(Xs)
        values_to_compare = [[2, 4], [1]]
        for i in range(len(missing_samples_by_mod)):
            assert pd.Index(missing_samples_by_mod[i]).equals(pd.Index(values_to_compare[i]))
        missing_samples_by_mod = get_missing_samples_by_mod(Xs, return_as_list=False)
        assert len(missing_samples_by_mod) == len(Xs)
        for i in range(len(missing_samples_by_mod)):
            assert pd.Index(missing_samples_by_mod[i]).equals(pd.Index(values_to_compare[i]))


def test_get_n_com_samples(sample_data):
    for Xs in sample_data[:2]:
        n_complete_samples = get_n_com_samples(Xs)
        assert n_complete_samples == 2


def test_get_n_incom_samples(sample_data):
    for Xs in sample_data[:2]:
        n_incomplete_samples = get_n_incom_samples(Xs)
        assert n_incomplete_samples == 3


def test_get_pct_com_samples(sample_data):
    for Xs in sample_data[:2]:
        percentage_complete = get_pct_com_samples(Xs)
        assert percentage_complete == (2 / 5) * 100


def test_get_pct_incom_samples(sample_data):
    for Xs in sample_data[:2]:
        percentage_incomplete = get_pct_incom_samples(Xs)
        assert percentage_incomplete == (3 / 5) * 100


if __name__ == "__main__":
    pytest.main()