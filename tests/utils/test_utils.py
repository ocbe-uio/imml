import pytest
rpy2 = pytest.importorskip("rpy2")
import importlib
import sys
from unittest.mock import patch
import pandas as pd

from imml.utils import _convert_df_to_r_object


def test_rmodule_installed():
    df = pd.DataFrame([[1, 2], [3, 4]])
    _convert_df_to_r_object(df)
    with patch.dict(sys.modules, {"rpy2": None}):
        import imml.utils.utils as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'r' needs to be installed to use r engine."):
            _convert_df_to_r_object(df)
    importlib.reload(module_mock)


if __name__ == "__main__":
    pytest.main()
