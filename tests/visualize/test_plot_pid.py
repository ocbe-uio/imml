import sys
import numpy as np
import pandas as pd
import pytest

from imml.visualize import plot_pid


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((30, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :2], X.iloc[:, 2:]
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    y_numpy = np.random.default_rng(42).choice(2, size=len(Xs_numpy[0]))
    y_pandas = pd.Series(y_numpy)
    return (Xs_pandas, y_pandas), (Xs_numpy, y_numpy)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_plot_pid_with_rus_abbreviations():
    rus = {"Uniqueness1": 1.2, "Uniqueness2": 0.8, "Redundancy": 0.5, "Synergy": 0.3}
    fig, ax = plot_pid(rus=rus, abb=True)
    assert fig is not None and ax is not None
    texts = [t.get_text() for t in ax.texts]
    assert any("U1\n" in t for t in texts)
    assert any("U2\n" in t for t in texts)
    assert any("R\n" in t for t in texts)
    assert any("S " in t for t in texts)
    assert any("1.2" in t for t in texts)
    assert any("0.8" in t for t in texts)
    assert any("0.5" in t for t in texts)
    assert any("0.3" in t for t in texts)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid rus."):
        rus = {"Redundancy": 0.2, "Synergy": 0.1}
        plot_pid(rus=rus, abb=False)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_plot_pid_with_Xs(sample_data):
    for Xs,y in sample_data:
        fig, ax = plot_pid(Xs=Xs, y=y, mod_names=["Mod1", "Mod2"], abb=False)
        assert fig is not None and ax is not None
        texts = [t.get_text() for t in ax.texts]
        assert any(t.startswith("Uniqueness\n") for t in texts)
        assert any(t.startswith("Redundancy\n") for t in texts)
        assert any("Synergy " in t for t in texts)
        assert any("Mod1" in t for t in texts)
        assert any("Mod2" in t for t in texts)


if __name__ == "__main__":
    pytest.main()