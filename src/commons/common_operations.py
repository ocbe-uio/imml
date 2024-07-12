import argparse
import itertools
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from imvc.datasets import LoadDataset

from settings import RANDOM_STATE, TIME_LIMIT
from src.utils.create_result_table import CreateResultTable
from src.clustering.run_clustering import RunClustering


class CommonOperations:


    @staticmethod
    def get_list_of_datasets(path):
        dataset_table = pd.read_csv(path)
        dataset_table = dataset_table.reindex(dataset_table.index.append(dataset_table.index[dataset_table["dataset"]=="nutrimouse"])).sort_index().reset_index(drop=True)
        dataset_table.loc[dataset_table["dataset"] == "nutrimouse", "dataset"] = ["nutrimouse_genotype", "nutrimouse_diet"]
        datasets = dataset_table["dataset"].to_list()
        two_view_datasets = dataset_table[dataset_table["n_features"].apply(lambda x: len(eval(x)) == 2)]["dataset"].to_list()
        return datasets, two_view_datasets


    @staticmethod
    def get_results_table(datasets, algorithms, probs, amputation_mechanisms, imputation, runs_per_alg, two_view_datasets):
        indexes_results = {"dataset": datasets, "algorithm": list(algorithms.keys()), "missing_percentage": probs,
                           "amputation_mechanism": amputation_mechanisms, "imputation": imputation, "run_n": runs_per_alg}
        indexes_names = list(indexes_results.keys())
        results = CreateResultTable.create_results_table(datasets=datasets, indexes_results=indexes_results,
                                                         indexes_names=indexes_names,
                                                         amputation_mechanisms=amputation_mechanisms,
                                                         two_view_datasets=two_view_datasets)
        return indexes_names, results


    @staticmethod
    def load_Xs_y(dataset_name):
        names = dataset_name.split("_")
        if "simulated" in names:
            names = ["_".join(names)]
        x_name,y_name = names if len(names) > 1 else (names[0], "0")
        Xs, y = LoadDataset.load_dataset(dataset_name=x_name, return_y=True)
        y = y[y_name]
        n_clusters = y.nunique()
        return Xs, y, n_clusters


    @staticmethod
    def run_processing(unfinished_results, dataset_name, indexes_names, results, algorithms,
                       incomplete_algorithms, subresults_path, logs_file, error_file, args,
                       results_path):

        Xs, y, n_clusters = CommonOperations.load_Xs_y(dataset_name=dataset_name)
        unfinished_results_dataset = unfinished_results.loc[[dataset_name]]

        if args.n_jobs == 1:
            iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)
            iterator.apply(
                lambda x: RunClustering.run_iteration(idx=x, results=results, Xs=Xs, y=y, n_clusters=n_clusters,
                                                      algorithms=algorithms,
                                                      incomplete_algorithms=incomplete_algorithms,
                                                      random_state=RANDOM_STATE,
                                                      subresults_path=subresults_path, logs_file=logs_file,
                                                      error_file=error_file), axis=1)
            results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=subresults_path,
                                                           indexes_names=indexes_names)

        else:
            if 0 in unfinished_results_dataset.index.get_level_values("missing_percentage"):
                unfinished_results_dataset_idx = unfinished_results_dataset.xs(0, level="missing_percentage",
                                                                               drop_level=False).index
                iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns=indexes_names)
                iterator.parallel_apply(lambda x: RunClustering.run_iteration(idx=x, results=results, Xs=Xs, y=y,
                                                                              n_clusters=n_clusters,
                                                                              algorithms=algorithms,
                                                                              incomplete_algorithms=incomplete_algorithms,
                                                                              random_state=RANDOM_STATE,
                                                                              subresults_path=subresults_path,
                                                                              logs_file=logs_file,
                                                                              error_file=error_file),
                                        axis=1)
                results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=subresults_path,
                                                               indexes_names=indexes_names)

                if args.save_results:
                    results.to_csv(results_path)

                unfinished_results_dataset_idx = unfinished_results_dataset.drop(unfinished_results_dataset_idx).index
                iterator = pd.DataFrame(unfinished_results_dataset_idx.to_list(), columns=indexes_names)
            else:
                iterator = pd.DataFrame(unfinished_results_dataset.index.to_list(), columns=indexes_names)

            iterator.parallel_apply(lambda x: RunClustering.run_iteration(idx=x, results=results, Xs=Xs, y=y,
                                                                          n_clusters=n_clusters,
                                                                          algorithms=algorithms,
                                                                          incomplete_algorithms=incomplete_algorithms,
                                                                          random_state=RANDOM_STATE,
                                                                          subresults_path=subresults_path,
                                                                          logs_file=logs_file,
                                                                          error_file=error_file), axis=1)
            results = CreateResultTable.collect_subresults(results=results.copy(), subresults_path=subresults_path,
                                                           indexes_names=indexes_names)
        if args.save_results:
            results.to_csv(results_path)
        shutil.rmtree(subresults_path)
        os.mkdir(subresults_path)

        return results


    @staticmethod
    def get_args():
        # args = lambda: None
        # args.continue_benchmarking, args.n_jobs, args.save_results = True, 2, False
        parser = argparse.ArgumentParser()
        parser.add_argument('-continue_benchmarking', default=False, action='store_true')
        parser.add_argument('-n_jobs', default=1, type=int)
        parser.add_argument('-save_results', default=False, action='store_true')
        args = parser.parse_args()
        return args


    @staticmethod
    def load_benchmarking(args, results, subresults_path, logs_file, error_file, results_path, indexes_names):
        if not args.continue_benchmarking:
            if not eval(input("Are you sure you want to start benchmarking and delete previous results? (True/False)")):
                raise Exception

            shutil.rmtree(subresults_path, ignore_errors=True)
            os.mkdir(subresults_path)
            results.to_csv(results_path)

            os.remove(logs_file) if os.path.exists(logs_file) else None
            os.remove(error_file) if os.path.exists(error_file) else None
            open(logs_file, 'w').close()
            open(error_file, 'w').close()

        else:
            finished_results = pd.read_csv(results_path, index_col=indexes_names)
            results.loc[finished_results.index, finished_results.columns] = finished_results
            finished_results = CreateResultTable.collect_subresults(results=results.copy(),
                                                                    subresults_path=subresults_path,
                                                                    indexes_names=indexes_names)
            results.loc[finished_results.index, finished_results.columns] = finished_results
        return results


    @staticmethod
    def limit_time(results, time_results_path, datasets, algorithms):
        results["time_limited"] = True
        time_results = pd.read_csv(time_results_path, index_col=0)
        for dataset_name, (alg_name, alg) in itertools.product(datasets, algorithms.items()):
            if (dataset_name in time_results.columns) and (alg_name in time_results.index):
                time_alg_dat = time_results.loc[alg_name, dataset_name]
                if (time_alg_dat > TIME_LIMIT) or (time_alg_dat <= 0) or np.isnan(time_alg_dat):
                    results.loc[(dataset_name, alg_name), "time_limited"] = False
        return results


    @staticmethod
    def get_unfinished_results(dataset_table_path, algorithms, probs, amputation_mechanisms, imputation, runs_per_alg,
                               args, subresults_path, logs_file, error_file, results_path, time_results_path,
                               incomplete_algorithms):
        datasets, two_view_datasets = CommonOperations.get_list_of_datasets(dataset_table_path)
        indexes_names, results = CommonOperations.get_results_table(datasets=datasets, algorithms=algorithms,
                                                                    probs=probs,
                                                                    amputation_mechanisms=amputation_mechanisms,
                                                                    imputation=imputation, runs_per_alg=runs_per_alg,
                                                                    two_view_datasets=two_view_datasets)
        results = CommonOperations.load_benchmarking(args=args, results=results, subresults_path=subresults_path,
                                                     logs_file=logs_file, error_file=error_file,
                                                     results_path=results_path,
                                                     indexes_names=indexes_names)
        results = CommonOperations.limit_time(results=results, time_results_path=time_results_path, datasets=datasets,
                                              algorithms=algorithms)

        if incomplete_algorithms:
            results = results.xs(False, level="imputation", drop_level=False)
        results = results.sort_index(level="missing_percentage", sort_remaining=False)
        unfinished_results = results.loc[~results["finished"] & results["time_limited"]]
        return indexes_names, results, unfinished_results


    @staticmethod
    def run_script(dataset_table_path, algorithms, probs, amputation_mechanisms, imputation, runs_per_alg, args,
                   subresults_path, logs_file, error_file, results_path, time_results_path,incomplete_algorithms):
        indexes_names, results, unfinished_results = CommonOperations.get_unfinished_results(
            dataset_table_path=dataset_table_path,
            algorithms=algorithms, probs=probs,
            amputation_mechanisms=amputation_mechanisms,
            imputation=imputation,
            runs_per_alg=runs_per_alg,
            args=args,
            subresults_path=subresults_path,
            logs_file=logs_file,
            error_file=error_file,
            results_path=results_path,
            time_results_path=time_results_path,
            incomplete_algorithms=incomplete_algorithms)

        for dataset_name in unfinished_results.index.get_level_values("dataset").unique():
            results = CommonOperations.run_processing(unfinished_results=unfinished_results, dataset_name=dataset_name,
                                                      indexes_names=indexes_names, results=results,
                                                      algorithms=algorithms,
                                                      incomplete_algorithms=incomplete_algorithms,
                                                      subresults_path=subresults_path,
                                                      logs_file=logs_file, error_file=error_file, args=args,
                                                      results_path=results_path)

        print("Completed successfully!")
        with open(logs_file, "a") as f:
            f.write(f'\n Completed successfully \t {datetime.now()}')





