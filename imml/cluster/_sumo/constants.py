# License: BSD-3-Clause

# command_line
__version__ = "0.3.0"
SUMO_COMMANDS = ["prepare", "run", "evaluate", "interpret"]

# prepare
CORR_METHODS = ["pearson", "spearman"]
SIMILARITY_METHODS = ["euclidean", "cosine"] + CORR_METHODS
TXT_EXT = [".txt", '.txt.gz', '.txt.bz2']
TSV_EXT = [".tsv", '.tsv.gz', '.tsv.bz2']
SUPPORTED_EXT = TXT_EXT + TSV_EXT

PREPARE_DEFAULTS = {
    "method": ["euclidean"],
    "k": 0.1,
    "alpha": 0.5,
    "missing": [0.1],
    "sn": 0,
    "fn": 0,
    "df": 0.1,
    "ds": 0.1,
    "plot": None,
    "atol": 1e-2
}
PREPARE_ARGS = ["infiles", "outfile"] + list(PREPARE_DEFAULTS.keys())  # 3 positional args

# run
CLUSTER_METHODS = ["max_value", "spectral"]
RUN_DEFAULTS = {
    "sparsity": [0.1],
    "n": 60,
    "method": "max_value",
    "max_iter": 500,
    "tol": 1e-5,
    "calc_cost": 20,
    "h_init": None,
    "t": 1,
    "subsample": 0.05,
    "rep": 5,
    "seed": None,
    "labels": None
}
RUN_ARGS = ["infile", "k", "outdir"] + list(RUN_DEFAULTS.keys())  # 3 positional args

# evaluate
EVALUATE_DEFAULTS = {
    "metric": None,
}
EVALUATE_ARGS = ["infile", "labels_file"] + list(EVALUATE_DEFAULTS.keys())  # 2 positional args

# interpret
INTERPRET_DEFAULTS = {
    "max_iter": 50,
    "n_folds": 5,
    "sn": 0,
    "fn": 0,
    "df": 0.1,
    "ds": 0.1,
    "t": 1,
    "seed": 1,
    "hits": 10
}
INTERPRET_ARGS = ['sumo_results', 'infiles', 'output_prefix'] + list(INTERPRET_DEFAULTS.keys())  # 3 positional args

# utils
CLUSTER_METRICS = ['NMI', 'purity', 'ARI']
COLOR_CODES = {
    "blue": {"light": "#3C548899", "dark": "#3C5488FF"},
    "red": {"light": "#DC000099", "dark": "#DC0000FF"},
    "green": {"light": "#00A08799", "dark": "#00A087FF"}
}
