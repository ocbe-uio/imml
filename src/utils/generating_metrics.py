import itertools
import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from reval.utils import kuhn_munkres_algorithm
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.multitest import multipletests
from tqdm.notebook import tqdm
from tqdm.contrib.itertools import product

from src.utils import GetMetrics


class ResultGenerator:


    @staticmethod
    def generate_results(results_path: str, supervised_metrics_path: str = None, inmetrics_path: str = None,
                         unsupervised_metrics_path: str = None, random_state = None,
                         n_permutations=1000, verbose=True, nb_workers=10, progress_bar=True):
        results = ResultGenerator.preprocess_results(results_path=results_path, nb_workers=nb_workers, verbose=verbose,
                                                     progress_bar=progress_bar)

        if unsupervised_metrics_path is not None:
            alg_uns_metrics = ResultGenerator.save_unsupervised_metrics(results=results,
                                                                        filepath=unsupervised_metrics_path,
                                                                        progress_bar=progress_bar)
        else:
            alg_uns_metrics = None
        if supervised_metrics_path is not None:
            supervised_metrics = ResultGenerator.save_supervised_metrics(results=results,
                                                                         filepath=supervised_metrics_path,
                                                                         random_state=random_state,
                                                                         n_permutations=n_permutations)
        else:
            supervised_metrics = None

        return results, alg_uns_metrics, supervised_metrics


    @staticmethod
    def save_alg_comparison(results: pd.DataFrame, filepath: str, progress_bar=True):
        comparison_columns = ["dataset", "alg1", "alg2", "run_n", "pred_alg1", "pred_alg2", "y_true"]
        alg_comparisons = pd.DataFrame([], columns=comparison_columns)

        if progress_bar:
            iterator = product(results["dataset"].unique(),
                    sorted(results["run_n"].unique()),
                    set(itertools.combinations(results["algorithm"].unique(), 2)))
        else:
            iterator = itertools.product(results["dataset"].unique(),
                    sorted(results["run_n"].unique()),
                    set(itertools.combinations(results["algorithm"].unique(), 2)))

        for dataset, run_n, (alg1, alg2) in iterator:
            pred_alg1 = results.loc[(results["dataset"] == dataset) &
                                    (results["missing_percentage"] == 0) &
                                    (results["run_n"] == run_n) &
                                    (results["algorithm"] == alg1),
            ["y_pred", "y_true", "y_pred_idx"]]
            pred_alg2 = results.loc[(results["dataset"] == dataset) &
                                    (results["missing_percentage"] == 0) &
                                    (results["run_n"] == run_n) &
                                    (results["algorithm"] == alg2),
            ["y_pred", "y_true", "y_pred_idx"]]
            if pred_alg1.empty or pred_alg2.empty:
                continue
            y1 = np.array(pred_alg1["y_true"].values[0])
            y2 = np.array(pred_alg2["y_true"].values[0])
            assert (y1 == y2).all()
            alg_comparison = pd.DataFrame([[dataset, alg1, alg2, run_n,
                                            pred_alg1["y_pred"].values[0], pred_alg2["y_pred"].values[0], y1]],
                                          columns=comparison_columns)
            alg_comparisons = pd.concat([alg_comparisons, alg_comparison], ignore_index=True)

        alg_comparisons["AMI"] = alg_comparisons.apply(
            lambda x: adjusted_mutual_info_score(x["pred_alg1"], x["pred_alg2"]), axis=1)
        alg_comparisons["ARI"] = alg_comparisons.apply(lambda x:
                                                       adjusted_rand_score(x["pred_alg1"], x["pred_alg2"]),
                                                       axis=1)
        alg_comparisons["Overlapping"] = alg_comparisons.apply(lambda x:
                                                               accuracy_score(x["pred_alg1"],
                                                                              kuhn_munkres_algorithm(
                                                                                  true_lab=x["pred_alg1"],
                                                                                  pred_lab=x["pred_alg2"])),
                                                               axis=1)
        alg_comparisons = alg_comparisons.groupby(
            ["alg1", "alg2"], as_index=False)[["AMI", "ARI", "Overlapping"]].mean().sort_values("AMI", ascending=False)
        alg_comparisons["Comparison"] = alg_comparisons["alg1"] + "_" + alg_comparisons["alg2"]
        alg_comparisons.to_csv(filepath, index=None)

        return alg_comparisons


    @staticmethod
    def save_unsupervised_metrics(results: pd.DataFrame, filepath: str, random_state=None, progress_bar=True):
        alg_stability = results[['dataset', 'algorithm', 'missing_percentage', 'amputation_mechanism', 'imputation', 'run_n',
                                 "y_pred", "y_pred_idx", 'silhouette', 'vrc', 'db', 'dbcv', 'dunn']]
        if alg_stability["imputation"].nunique() == 1:
            alg_stability = alg_stability.loc[alg_stability["missing_percentage"] == 0]
        else:
            alg_stability = alg_stability.loc[
                (alg_stability["missing_percentage"] == 0) | (alg_stability["imputation"])
                ]
        alg_uns_metrics = alg_stability.drop(columns=["y_pred", "y_pred_idx", 'imputation', 'run_n'])
        alg_uns_metrics = alg_uns_metrics.groupby(
            ["dataset", "algorithm", "missing_percentage", "amputation_mechanism"], as_index=False).mean()
        alg_stability["sorted_preds"] = alg_stability.apply(
            lambda x: pd.Series(x["y_pred"], index=x["y_pred_idx"]).sort_index().to_list(), axis=1)

        iterator = alg_stability["dataset"].unique()
        if progress_bar:
            iterator = tqdm(iterator)

        for dataset in iterator:
            preds_dataset = alg_stability.loc[
                (alg_stability["dataset"] == dataset), ["missing_percentage", "algorithm", 'amputation_mechanism',
                                                        "run_n", "sorted_preds"]]
            for alg in preds_dataset["algorithm"].unique():
                pred_alg = preds_dataset[preds_dataset["algorithm"] == alg]
                for missing_percentage in pred_alg["missing_percentage"].unique():
                    pred_missing_alg = pred_alg[pred_alg["missing_percentage"] == missing_percentage]
                    for amputation_mechanism in pred_missing_alg["amputation_mechanism"].unique():
                        pred_missing_ampt_alg = pred_missing_alg[
                            pred_missing_alg["amputation_mechanism"] == amputation_mechanism]
                        amis, aris = [], []
                        for run_1, run_2 in set(itertools.combinations(pred_missing_ampt_alg["run_n"].unique(), 2)):
                            pred1_alg = pred_missing_ampt_alg.loc[
                                (pred_missing_ampt_alg["run_n"] == run_1), "sorted_preds"].to_list()[0]
                            pred2_alg = pred_missing_ampt_alg.loc[
                                (pred_missing_ampt_alg["run_n"] == run_2), "sorted_preds"].to_list()[0]
                            amis.append(adjusted_mutual_info_score(pred1_alg, pred2_alg)), aris.append(
                                adjusted_rand_score(pred1_alg, pred2_alg))

                        alg_uns_metrics.loc[(alg_uns_metrics["dataset"] == dataset) &
                                            (alg_uns_metrics["missing_percentage"] == missing_percentage) &
                                            (alg_uns_metrics["amputation_mechanism"] == amputation_mechanism) &
                                            (alg_uns_metrics["algorithm"] == alg),
                        ["AMI", "ARI"]] = [np.mean(amis), np.mean(aris)]

        alg_uns_metrics.to_csv(filepath, index=None)
        return alg_uns_metrics


    @staticmethod
    def preprocess_results(results_path: str, verbose=True, nb_workers=10, progress_bar=True):
        pandarallel.initialize(nb_workers=nb_workers, progress_bar=progress_bar)
        results = pd.read_csv(results_path)
        if verbose:
            print("results", results.shape)
        results = results[results["finished"]]
        if verbose:
            print("finished_results", results.shape)
        results = results[results["completed"]]
        if verbose:
            print("completed_results", results.shape)
        results[["y_true", "y_pred", "y_true_idx", "y_pred_idx"]] = results[
            ["y_true", "y_pred", "y_true_idx", "y_pred_idx"]].parallel_applymap(eval)
        assert results["y_true_idx"].eq(results["y_pred_idx"]).all()
        return results


    @staticmethod
    def save_supervised_metrics(results: pd.DataFrame, filepath: str, random_state=None, n_permutations: int = 1000):
        supervised_metrics = results[["y_true", "y_pred"]].parallel_apply(
            lambda row: GetMetrics.compute_supervised_metrics(y_true=row["y_true"], y_pred=row["y_pred"],
                                                              random_state=random_state, n_permutations=n_permutations),
            axis=1)
        results = pd.concat([results, pd.DataFrame(supervised_metrics.to_dict()).T], axis=1)
        indexes_names = ["dataset", "algorithm", "missing_percentage", "amputation_mechanism", "imputation"]
        results = results[results.select_dtypes(include="float").columns.to_list() + indexes_names].groupby(
            indexes_names, sort=False).agg(["mean", 'std']).reset_index()
        results.columns = results.columns.map('_'.join).str.strip('_')
        results["padj"] = multipletests(results["MCC (p-value)_mean"], method="fdr_bh")[1]
        results["log_padj"] = results["padj"].apply(lambda x: -np.log10(x))
        results.to_csv(filepath, index=None)
        return results


    # def save_stability_metrics(results: pd.DataFrame, filepath: str, random_state=None, progress_bar=True):
    #     results = pd.merge(results, pd.DataFrame(itertools.product(results["dataset"].unique(), results["algorithm"].unique()),
    #                                     columns=["dataset", "algorithm"]), how="right")
    #     res = OneHotEncoder(sparse_output=False).set_output(transform="pandas").fit_transform(
    #         results[["dataset", "algorithm"]])
    #     for col in ["silhouette_mean", "silhouette_std", "MCC_mean", "MCC_std", "MCC (p-value)_mean",
    #                 "MCC (p-value)_std"]:
    #         res[col] = results[col]
    #         results[col] = KNNImputer().set_output(transform="pandas").fit_transform(X=res)[col]
    #         res = res.drop(columns=col)
    #     results["padj"] = multipletests(results["MCC (p-value)_mean"], method="fdr_bh")[1]
    #     results["log_padj"] = results["padj"].apply(lambda x: -np.log10(x))
    #     results.to_csv(inmetrics_path, index=None)
    #     outputs.append(results)
    #     return outputs



