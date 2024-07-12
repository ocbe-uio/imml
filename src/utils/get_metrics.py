import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
from reval.utils import kuhn_munkres_algorithm
from permetrics import ClusteringMetric


class GetMetrics:


    @staticmethod
    def compute_supervised_metrics(y_true, y_pred, random_state = None, n_permutations=1000):
        random_preds = [pd.Series(y_true).value_counts().index[0]] * len(y_true)
        perm_clust_labels = kuhn_munkres_algorithm(true_lab=y_true, pred_lab=y_pred)
        mcc, p_value = GetMetrics.compute_mcc(y_true=y_true, y_pred=perm_clust_labels, random_state=random_state,
                                              n_permutations=n_permutations)
        supervised_metrics = {
            'ACC': metrics.accuracy_score(y_true=y_true, y_pred=perm_clust_labels),
            'MCC': mcc,
            'MCC (p-value)': p_value,
            'F1': metrics.f1_score(y_true=y_true, y_pred=perm_clust_labels, average='macro'),
            'precision': metrics.precision_score(y_true=y_true, y_pred=perm_clust_labels, average='macro', zero_division=0),
            'recall': metrics.recall_score(y_true=y_true, y_pred=perm_clust_labels, average='macro', zero_division=0),
            "bal_acc": metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
            "ami": metrics.adjusted_mutual_info_score(labels_true=y_true, labels_pred=y_pred),
            "ari": metrics.adjusted_rand_score(labels_true=y_true, labels_pred=y_pred),
            "completeness": metrics.completeness_score(labels_true=y_true, labels_pred=y_pred),
            "random_acc": metrics.accuracy_score(y_true=y_true, y_pred=random_preds),
            "random_f1": metrics.f1_score(y_true=y_true, y_pred=random_preds, average='macro'),
        }
        return supervised_metrics


    @staticmethod
    def compute_unsupervised_metrics(X, y_pred, random_state = None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        unsupervised_metrics = {}
        try: unsupervised_metrics["silhouette"] = metrics.silhouette_score(X = X, labels = y_pred, random_state= random_state)
        except: unsupervised_metrics["silhouette"] = np.nan
        try: unsupervised_metrics["vrc"] = metrics.calinski_harabasz_score(X = X, labels = y_pred)
        except: unsupervised_metrics["vrc"] = np.nan
        try: unsupervised_metrics["db"] = metrics.davies_bouldin_score(X = X, labels = y_pred)
        except:unsupervised_metrics["db"] = np.nan
        met = ClusteringMetric(X=X, y_pred=y_pred)
        try: unsupervised_metrics["dbcv"] = met.DBCVI()
        except:unsupervised_metrics["dbcv"] = np.nan
        try: unsupervised_metrics["dunn"] = met.DI()
        except:unsupervised_metrics["dunn"] = np.nan
        try: unsupervised_metrics["dhi"] = met.DHI()
        except:unsupervised_metrics["dhi"] = np.nan
        try: unsupervised_metrics["ssei"] = met.SSEI()
        except:unsupervised_metrics["ssei"] = np.nan
        try: unsupervised_metrics["rsi"] = met.RSI()
        except:unsupervised_metrics["rsi"] = np.nan
        try: unsupervised_metrics["bhi"] = met.BHI()
        except:unsupervised_metrics["bhi"] = np.nan

        return unsupervised_metrics


    @staticmethod
    def compute_mcc(y_true, y_pred, n_permutations=1000, random_state=None):
        mcc_value = metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        if random_state is not None:
            p_values = [metrics.matthews_corrcoef(y_true=shuffle(y_true, random_state=i+random_state),
                                                  y_pred=shuffle(y_pred, random_state=i))
                        for i in range(n_permutations)]
        else:
            p_values = [metrics.matthews_corrcoef(y_true=shuffle(y_true),
                                                  y_pred=shuffle(y_pred))
                        for i in range(n_permutations)]
        p_values = (pd.Series(p_values) > mcc_value).sum() / len(p_values)
        if p_values == 0:
            p_values = 1/(n_permutations + 1)
        return mcc_value, p_values
