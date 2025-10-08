# License: BSD-3-Clause

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    base = importr('base')
    rmodule_installed = True
except ImportError:
    rmodule_installed = False
    rmodule_error = "Module 'r' needs to be installed to use r engine. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"


def _convert_df_to_r_object(dataframe):
    if not rmodule_installed:
        raise ImportError(rmodule_error)
    with (ro.default_converter + pandas2ri.converter).context():
        r_from_pd_df = ro.conversion.get_conversion().py2rpy(dataframe)
    return base.lapply(r_from_pd_df, base.as_matrix)
