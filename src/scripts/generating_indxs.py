import argparse
import itertools
import json
import os.path
import shutil
import numpy as np
from sklearn.utils import shuffle
from imvc.ampute import Amputer
from imvc.impute import get_observed_view_indicator

from settings import PROFILES_PATH, DATASET_TABLE_PATH, RANDOM_STATE, probs, amputation_mechanisms, runs_per_alg
from src.commons import CommonOperations

parser = argparse.ArgumentParser()
parser.add_argument('-continue_indxs', default=False, action='store_true')
parser.add_argument('-save_results', default=False, action='store_true')
args = parser.parse_args()

if not args.continue_indxs:
    shutil.rmtree(PROFILES_PATH, ignore_errors=True)
    os.mkdir(PROFILES_PATH)

datasets, two_view_datasets = CommonOperations.get_list_of_datasets(DATASET_TABLE_PATH)

for dataset_name in datasets:
    Xs, y, n_clusters = CommonOperations.load_Xs_y(dataset_name=dataset_name)

    for prob, amputation_mechanism, run_n in itertools.product(probs, amputation_mechanisms, runs_per_alg):
        if prob == 0:
            if amputation_mechanism == "EDM":
                amputation_mechanism = "No"
            else:
                continue

        path = f"{dataset_name}_{prob}_{amputation_mechanism}_{run_n}.json"
        path = os.path.join(PROFILES_PATH, path)
        if os.path.exists(path):
            continue
        try:
            random_state = RANDOM_STATE + run_n
            if (dataset_name in two_view_datasets) and (amputation_mechanism in ["MAR", "MNAR"]):
                continue
            *train_Xs, y_train = shuffle(*Xs, y, random_state=random_state)
            p = prob/100

            if p != 0:
                amp = Amputer(p=round(p, 2), mechanism=amputation_mechanism, random_state=random_state)
                train_Xs = amp.fit_transform(train_Xs)

            observed_view_indicator = get_observed_view_indicator(train_Xs)
            try:
                lower_index = observed_view_indicator.index.astype(np.int16)
                assert (train_Xs[0].index == lower_index).all()
                observed_view_indicator.index = lower_index
            except AssertionError:
                assert (train_Xs[0].index == observed_view_indicator.index).all()

            dict_indxs = {
                "observed_view_indicator": observed_view_indicator.to_dict(),
                "valid": True,
            }

        except (AssertionError, ValueError) as exception:
            dict_indxs = {
                "valid" : False,
                "error": str(exception),
            }

        if args.save_results:
            with open(path, "w") as f:
                json.dump(dict_indxs, f)

print("Completed successfully!")


