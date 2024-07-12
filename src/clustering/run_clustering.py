import json
import os
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imvc.preprocessing import MultiViewTransformer, ConcatenateViews
from imvc.utils import DatasetUtils
from settings import PROFILES_PATH

from src.models import Model
from src.utils import GetMetrics


class RunClustering:


    @staticmethod
    def run_iteration(idx, results, Xs, y, n_clusters, algorithms, incomplete_algorithms,
                      random_state, subresults_path, logs_file, error_file):
        errors_dict = {}
        row = results.loc[[idx]]
        try:
            with open(logs_file, "a") as f:
                f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t {datetime.now()}')
            row_index = row.index
            dataset_name, alg_name, p, amputation_mechanism, impute, run_n = (
                row_index.get_level_values("dataset")[0],
                row_index.get_level_values("algorithm")[0],
                row_index.get_level_values("missing_percentage")[0],
                row_index.get_level_values("amputation_mechanism")[0],
                row_index.get_level_values("imputation")[0],
                row_index.get_level_values("run_n")[0])
            alg = algorithms[alg_name]

            path = f"{dataset_name}_{p}_{amputation_mechanism}_{run_n}.json"
            path = os.path.join(PROFILES_PATH, path)
            with open(path) as f:
                observed_view_indicator = json.load(f)
                if observed_view_indicator["valid"]:
                    observed_view_indicator = observed_view_indicator["observed_view_indicator"]
                else:
                    raise ValueError(observed_view_indicator["error"])
            observed_view_indicator = pd.DataFrame.from_dict(observed_view_indicator)
            observed_view_indicator.index = observed_view_indicator.index.astype(int)
            observed_view_indicator.columns = observed_view_indicator.columns.astype(int)
            train_Xs = DatasetUtils.convert_to_imvd(Xs=Xs, observed_view_indicator=observed_view_indicator)
            train_Xs = [X.loc[observed_view_indicator.index] for X in train_Xs]
            y_train = y.loc[train_Xs[0].index]

            if impute:
                train_Xs = MultiViewTransformer(SimpleImputer(strategy="mean").set_output(
                    transform="pandas")).fit_transform(train_Xs)
            elif not incomplete_algorithms:
                train_Xs = DatasetUtils.select_complete_samples(Xs=train_Xs)
                y_train = y_train.loc[train_Xs[0].index]
                try:
                    assert len(y_train) > n_clusters
                except AssertionError:
                    raise ValueError("Number of clusters is lower than the number of complete samples.")

            start_time = time.perf_counter()
            clusters, train_X = Model(alg_name=alg_name, alg=alg).method(train_Xs=train_Xs, n_clusters=n_clusters,
                                                                         random_state=random_state, run_n=run_n)
            elapsed_time = time.perf_counter() - start_time
            clusters = pd.Series(clusters, index=y_train.index)

            if isinstance(train_X, list):
                train_X = ConcatenateViews().fit_transform(train_X)
            if np.isnan(train_X).any().any():
                train_X = SimpleImputer(strategy="mean").fit_transform(train_X)
            if not isinstance(train_X, pd.DataFrame):
                train_X = pd.DataFrame(train_X, index=y_train.index)

            assert train_X.index.equals(y_train.index)
            assert train_Xs[0].index.equals(y_train.index)
            assert clusters.index.equals(y_train.index)

            dict_results = RunClustering.save_record(train_Xs=train_Xs, train_X=train_X, y_pred=clusters, y_true=y_train,
                                                     elapsed_time=elapsed_time, random_state=random_state,
                                                     errors_dict=errors_dict)
            dict_results = pd.DataFrame(pd.Series(dict_results), columns=row_index).T
            row[dict_results.columns] = dict_results
            row[["finished", "completed"]] = True
        except Exception as exception:
            exception_name = type(exception).__name__
            errors_dict[f"{exception_name}"] = exception
            row[["finished", "comments"]] = True, [dict(errors_dict)]
            with open(error_file, "a") as f:
                f.write(f'\n {row.drop(columns=row.columns).reset_index().to_dict(orient="records")[0]} \t'
                        f'  {exception_name}: {exception}  \t {datetime.now()}')

        row.to_csv(os.path.join(subresults_path, f"{'_'.join([str(i) for i in idx])}.csv"))
        return row


    @staticmethod
    def save_record(train_Xs, train_X, y_pred, y_true, elapsed_time, random_state, errors_dict):
        missing_clusters_mask = np.invert(np.isnan(y_pred))

        dict_results = {
            "n_samples": len(train_X),
            "n_views": len(train_Xs),
            "n_incomplete_samples": DatasetUtils.get_n_incomplete_samples(train_Xs),
            "n_complete_samples": DatasetUtils.get_n_complete_samples(train_Xs),
            "time": elapsed_time,
            "n_clustered_samples": missing_clusters_mask.sum(),
            "percentage_clustered_samples": 100* missing_clusters_mask.sum() // len(train_X),
            "comments": dict(errors_dict),
            "label_sizes": y_true.value_counts().to_dict(),
            "cluster_sizes": y_pred.value_counts(dropna=False).to_dict(),
            "relative_label_sizes": y_true.value_counts(normalize=True).to_dict(),
            "relative_cluster_sizes": y_pred.value_counts(normalize=True, dropna=False).to_dict(),
            "y_true": y_true.to_list(),
            "y_pred": y_pred.to_list(),
            "y_true_idx": y_true.index.to_list(),
            "y_pred_idx": y_pred.index.to_list(),
            **GetMetrics.compute_unsupervised_metrics(X=train_X, y_pred=y_pred, random_state=random_state)
        }

        if not all(missing_clusters_mask):
            clusters_excluding_missing = y_pred[missing_clusters_mask].astype(int)
            X_excluding_missing = train_X.iloc[missing_clusters_mask]
            clusters_excluding_missing = pd.factorize(clusters_excluding_missing)[0]

            summary_results = GetMetrics.compute_unsupervised_metrics(y_pred=clusters_excluding_missing,
                                                    X=X_excluding_missing, random_state=random_state)
            summary_results = {f"{key}_excluding_outliers": value for key, value in summary_results.items()}
            dict_results = {**dict_results, **summary_results}
            y_pred[~missing_clusters_mask] = -1

        dict_results = {**dict_results,
                        **GetMetrics.compute_unsupervised_metrics(X=train_X, y_pred=y_pred, random_state=random_state)}

        return dict_results
