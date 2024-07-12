import itertools
import os
import shutil
import numpy as np
import pandas as pd


class CreateResultTable:


    @staticmethod
    def create_results_table(datasets, indexes_results, indexes_names, amputation_mechanisms, two_view_datasets):
        # create df with all datasets
        results = pd.DataFrame(datasets, columns=["dataset"])
        # merge result df with all experimental options
        for k, v in {k: v for k, v in indexes_results.items() if k != "dataset"}.items():
            results = results.merge(pd.Series(v, name=k), how="cross")
        # change the name when there is no missing
        results.loc[(results["amputation_mechanism"] == "EDM") & (
                results["missing_percentage"] == 0), "amputation_mechanism"] = "No"
        results = results.set_index(indexes_names)

        # remove experiments when there is no missing and imputation is True
        idx_to_drop = results.xs(0, level="missing_percentage",
                                 drop_level=False).xs(True, level="imputation", drop_level=False).index
        results = results.drop(idx_to_drop)
        # remove experiments when missing percentage is 0 and there is amputation
        for amputation_mechanism in amputation_mechanisms[1:]:
            idx_to_drop = results.xs(0, level="missing_percentage",
                                     drop_level=False).xs(amputation_mechanism, level="amputation_mechanism",
                                                          drop_level=False).index
            results = results.drop(idx_to_drop)
        # keep only one experiment when there is no missing
        results_amputation_mechanism_none = results.xs(0, level="missing_percentage", drop_level=False)
        results_amputation_mechanism_none_tochange = results_amputation_mechanism_none.index.to_frame()
        results_amputation_mechanism_none_tochange["amputation_mechanism"] = "'None'"
        results.loc[results_amputation_mechanism_none.index].index = pd.MultiIndex.from_frame(
            results_amputation_mechanism_none_tochange)

        # when there are only two views, remove amputation for multiple views
        for amputation_mechanism, dataset in itertools.product(["MAR", "MNAR"], two_view_datasets):
            idx_to_drop = results.xs(dataset, level="dataset",
                                     drop_level=False).xs(amputation_mechanism, level="amputation_mechanism",
                                                          drop_level=False).index
            results = results.drop(idx_to_drop)

        results[["finished", "completed"]] = False
        return results


    @staticmethod
    def collect_subresults(results, subresults_path, indexes_names):
        # get paths to all subresult files
        if os.path.exists(subresults_path):
            subresults_files = pd.Series(os.listdir(subresults_path)).apply(lambda x: os.path.join(subresults_path, x))
            # filter folders
            subresults_files = subresults_files[subresults_files.apply(os.path.isfile)]
            # if there are files
            if len(subresults_files) > 0:
                # read files and concat them
                subresults_files = pd.concat(subresults_files.apply(pd.read_csv).to_list())
                # put same format as result df
                subresults_files = subresults_files.set_index(indexes_names)
                # include only those finished
                subresults_files = subresults_files[subresults_files["finished"]]
                # add them to our result df
                results.loc[subresults_files.index, subresults_files.columns] = subresults_files
            drop_columns = "comments"
            # fix nan values
            results_ = results.select_dtypes(object).drop(columns=drop_columns).replace(np.nan, "np.nan")
            try:
                results[results_.columns] = results_.parallel_applymap(lambda x: eval(str(x)))
            except:
                results[results_.columns] = results_.applymap(lambda x: eval(str(x)))
        return results


    @staticmethod
    def remove_subresults(subresults_path):
        shutil.rmtree(subresults_path)
        os.mkdir(subresults_path)
        return None
