# License: BSD-3-Clause

import operator
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin

from ._monet._aux_monet import _best_samples_to_add, _which_sample_to_remove, _which_view_to_add_to_module, \
    _which_view_to_remove_from_module, _score_of_split_module, _weight_of_split_and_add_view, \
    _weight_of_split_and_remove_view, _weight_of_new_module, _top_samples_to_switch, \
    _weight_of_spreading_module, _weight_of_merged_modules, _Globals, _Sample, _Module, _View, _switch_2_samples
from ..utils import check_Xs
from ..preprocessing import remove_missing_samples_by_mod


class MONET(BaseEstimator, ClassifierMixin):
    r"""
    Multi Omic Clustering by Non-Exhaustive Types (MONET). [#monetpaper]_ [#monetcode]_

    MONET operates in two distinct phases to extract meaningful information from multi-omics datasets. In the first
    phase, it constructs an edge-weighted graph for each omic, where the nodes represent individual samples, and the
    weights indicate the similarity between samples within that particular omic. Moving on to the second phase, MONET
    identifies modules by identifying dense subgraphs that are shared across multiple omic graphs.

    The resulting output comprises a collection of modules, each representing a subset of samples. These modules are
    mutually exclusive, meaning that samples are assigned to only one module. It is important to note that not all
    samples are necessarily assigned to a module; those remaining unassigned are referred to as "lonely" samples.
    Each module, denoted as M, is characterized by its constituent samples, referred to as samples(M), and the set of
    omics it encompasses, denoted as omics(M). Intuitively, samples(M) exhibit similarity with one another
    specifically within the omics(M) context.

    Parameters
    ----------
    n_clusters : Ignored
        Ignored.
    num_repeats : int (default=15)
        Times the algorithm will be repeated in order to avoid suboptimal (local maximum) solutions. The best solution
        will be returned.
    similarity_mode : str (default='prob')
        One of ['prob', 'corr']. If 'corr', the weighting scheme is computed basen on correlation; if 'prob',
        a probabilistic formulation is used.
    init_modules : dict (default=None)
        an optional module initialization for MONET. A dict mapping between module names to sample ids. All modules
        are initialized to cover all views. Set to None to use MONET's seed finding algorithm for initialization.
    iters : int (default=500)
        Maximal number of iterations.
    num_of_seeds : int (default=10)
        Number of seeds to create in MONET's module initialization algorithm.
    num_of_samples_in_seed : int (default=10)
        Number of samples to put in each seeds to create in MONET's module initialization algorithm.
    min_mod_size : int (default=10)
        Minimal size (number of samples) for a MONET module.
    max_samples_per_action : int (default=10)
        Maximal number of samples in a single MONET action (maximal number of samples added to a module or replaced
        between modules in a single action).
    percentile_remove_edge : int (default=None)
        Only edges with weight percentile above (for positive weights) or below (for negative weights) this percentile
        are kept in the graph. For example, percentile_remove_edge=90 keeps only the 10% edges with highest positive
        weight and lowest negative weight in the graph. one keeps all edges in the graph.
    random_state : int (default=None)
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.
    n_jobs : int (default=None)
        The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    glob_var_ : dict
        Module names to Module objects mapping. Every module instance includes its set of
        samples (under the "samples" attribute) and its set of views (the "views" attribute).
    total_weight_ : float
        Sum of the weights (similarity between samples within the module) of all modules.
    mod_graphs_ : list of dataframes of shape (n_samples, n_samples)
        Graph of each modality.
    mod_views_ : list of length n_mods.
        Views used for each module.
    n_clusters_ : int
        Number of clusters.

    References
    ----------
    .. [#monetpaper] Rappoport N, Safra R, Shamir R (2020) MONET: Multi-omic module discovery by omic selection. PLoS
                     Comput Biol 16(9): e1008182. https://doi.org/10.1371/journal.pcbi.1008182.
    .. [#monetcode] https://github.com/Shamir-Lab/MONET

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import MONET
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = MONET()
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = None, num_repeats: int = 15, similarity_mode: str = 'corr',
                 init_modules: dict = None, iters: int = 500, num_of_seeds: int = 10,
                 num_of_samples_in_seed: int = 10, min_mod_size: int = 10, max_sams_per_action: int = 10,
                 percentile_remove_edge: int = None, random_state: int = None,
                 verbose: bool = False, n_jobs: int = None):
        similarity_mode_opts = ["corr"]
        if similarity_mode not in similarity_mode_opts:
            raise ValueError(f"Invalid similarity_mode. Expected one of {similarity_mode_opts}. {similarity_mode} was passed.")

        self.n_clusters = n_clusters
        self.num_repeats = num_repeats
        self.similarity_mode = similarity_mode
        self.init_modules = init_modules
        self.iters = iters
        self.num_of_seeds = num_of_seeds
        self.num_of_samples_in_seed = num_of_samples_in_seed
        self.min_mod_size = min_mod_size
        self.max_sams_per_action = max_sams_per_action
        self.percentile_remove_edge = percentile_remove_edge
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        # a list of the actions considered by MONET in each iteration. Each action correponds to one function in the list.
        self.functions = [_best_samples_to_add, _which_sample_to_remove, _which_view_to_add_to_module,
                          _which_view_to_remove_from_module, _score_of_split_module, _weight_of_split_and_add_view,
                          _weight_of_split_and_remove_view, _weight_of_new_module, _top_samples_to_switch,
                          _weight_of_spreading_module, _weight_of_merged_modules]


    def fit(self, Xs, y=None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different views.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns an instance of self.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]
        for X in Xs:
            X.index = X.index.astype(str)
        samples = Xs[0].index
        Xs = remove_missing_samples_by_mod(Xs=Xs)
        data = {}

        if self.similarity_mode == "corr":
            data = self._process_data(Xs=Xs)
        solutions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(
                data=data, init_modules=self.init_modules, iters=self.iters, num_of_seeds=self.num_of_seeds,
                num_of_samples_in_seed=self.num_of_samples_in_seed, min_mod_size=self.min_mod_size,
                max_sams_per_action=self.max_sams_per_action, percentile_remove_edge=self.percentile_remove_edge,
                samples = samples, verbose=self.verbose,
                random_state=self.random_state + n_time if self.random_state is not None else self.random_state)
            for n_time in range(self.num_repeats)
        )
        solutions = {idx: i for idx, i in enumerate(solutions)}
        best_sol = {key: value['total_weight'] for key, value in solutions.items()}
        best_sol = max(best_sol.items(), key=operator.itemgetter(1))[0]
        best_sol = solutions[best_sol]
        glob_var, total_weight = best_sol['glob_var'], best_sol['total_weight']
        labels, view_graphs, mod_views = self._post_processing(glob_var=glob_var)
        labels = labels.loc[samples].squeeze().values
        labels_wo_nan = np.unique(labels, return_inverse=True)[1].astype(float)
        labels_wo_nan[np.isnan(labels)] = np.nan
        self.labels_ = labels_wo_nan
        self.glob_var_ = glob_var
        self.total_weight_ = total_weight
        self.view_graphs_ = view_graphs
        self.mod_views_ = mod_views
        self.n_clusters_ = len(np.unique(labels_wo_nan[~np.isnan(labels_wo_nan)]))
        return self


    def _predict(self, Xs):
        r"""
        Return clustering results for samples.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different views.

        Returns
        -------
        labels : list of array-likes objects, shape (n_samples,)
            The predicted data.
        """
        labels = self.labels_
        return labels


    def fit_predict(self, Xs, y=None):
        r"""
        Fit the model and return clustering results.
        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different views.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            The predicted data.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels


    def _single_run(self, data, init_modules, iters, num_of_seeds, num_of_samples_in_seed, min_mod_size,
                    max_sams_per_action, percentile_remove_edge, samples, random_state, verbose):
        r"""

        """
        if random_state is None:
            random_state = np.random.default_rng().integers(100000)
        glob_var = _Globals(len(self.functions))
        glob_var = self._create_env(samples = samples, glob_var=glob_var, data=data,
                                    percentile_remove_edge=percentile_remove_edge)
        glob_var.min_mod_size = min_mod_size
        glob_var.max_samps_per_action = max_sams_per_action

        if init_modules is None:
            glob_var = self._get_seeds(glob_var, num_of_seeds=num_of_seeds,
                                       num_of_samples_in_seed=num_of_samples_in_seed, random_state=random_state)
        else:
            glob_var = self._create_seeds_from_solution(glob_var, init_modules)

        for some_mod in glob_var.modules.copy().values():
            if len(some_mod.samples) < min_mod_size:
                if verbose:
                    print('killing a small module before starting')
                glob_var.kill_module(some_mod)

        total_weight = sum(mod.get_weight() for mod in glob_var.modules.values())
        converged_modules = {}
        did_action = False
        iterations = 0

        while iterations < iters:
            prev_weight = total_weight

            active_module_names = list(sorted(set(glob_var.modules.keys()) - set(converged_modules.keys())))
            if len(active_module_names) == 0:
                if not did_action:
                    if verbose:
                        print("converged, total score: {}.".format(total_weight))
                    break
                else:
                    converged_modules = {}
                    did_action = False
                    active_module_names = list(sorted(glob_var.modules.keys()))
            mod_name = np.random.default_rng(random_state + iterations).choice(active_module_names).tolist()
            mod = glob_var.modules[mod_name]

            max_res = self._get_next_step(mod, glob_var)
            glob_var = self._exec_next_step(mod, max_res, glob_var)
            for _, some_mod in glob_var.modules.copy().items():
                if len(some_mod.get_samples()) <= 1 or not some_mod.get_views():
                    glob_var.kill_module(some_mod)
                    if verbose:
                        print('removing zombie module')

            total_weight = sum([mod.get_weight() for name, mod in glob_var.modules.items()])
            iterations += 1

            if (iterations % 10 == 0) and verbose:
                print("iteration: " + str(iterations))
                print("num of modules: " + str(len(glob_var.modules)))
                print("total_weight: " + str(total_weight))
                print("actions: " + str(glob_var.actions))

            # Assert module sizes
            for _, some_mod in glob_var.modules.copy().items():
                assert len(some_mod.samples) >= min_mod_size

            if total_weight <= prev_weight or max_res[1][0] == -float("inf"):
                if mod_name in glob_var.modules:
                    converged_modules.update({mod_name: glob_var.modules[mod_name]})
            else:  # the score deviates from the score we expected
                if not (abs(total_weight - prev_weight - max_res[1][0]) < 0.01):
                    # This signifies a bug and should never occur:
                    # that the difference in the objective function from the
                    # previous iteration is different from the difference
                    # the algorithm expected for the function.
                    raise Exception("The clusters could not be found.")
                did_action = True
                assert abs(total_weight - prev_weight - max_res[1][0]) < 0.01
                did_action = True
        for mod_name, mod in glob_var.modules.copy().items():
            if mod.get_size() <= glob_var.min_mod_size and not self._is_mod_significant(mod, glob_var,
                                                                                        random_state=random_state):
                if verbose:
                    print("module {} with samples {} on views {} is not significant.".format((mod_name, mod),
                                                                                             mod.get_samples(),
                                                                                             mod.get_views().keys()))
                glob_var.kill_module(mod)

        return {"glob_var": glob_var, "total_weight": total_weight}


    @staticmethod
    def _process_data(Xs: list):
        """gets raw data and return a list of similarity matrices"""
        data = {}
        for X_idx, X in enumerate(Xs):
            X_t = X.copy().T
            X_t.columns = X_t.columns
            X_t = X_t.corr()
            np.fill_diagonal(X_t.values, 0)
            data[str(X_idx)] = X_t
        return data


    def _create_env(self, samples, glob_var, data, percentile_remove_edge):
        """
        Create all the variables used during MONET's run:
        modules, modality, etc, and associating them with a Global instance.
        """
        all_sam_names = set(samples)
        glob_var.samples = {sample: _Sample(sample) for sample in all_sam_names}

        for view, dat in data.items():
            self.view = view
            graph, means, covs, percentile = self._build_a_graph_similarity(dat)
            if percentile_remove_edge is not None:
                all_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
                all_weights_array = np.array(all_weights)
                positive_thresh = np.percentile(all_weights_array[all_weights_array > 0], percentile_remove_edge)
                negative_thresh = np.percentile(all_weights_array[all_weights_array < 0], 100 - percentile_remove_edge)
                all_edges = [edge for edge in graph.edges]
                for edge in all_edges:
                    cur_weight = graph.edges[edge]['weight']
                    if (cur_weight > 0 and cur_weight < positive_thresh) or (cur_weight < 0 and cur_weight > negative_thresh):
                        graph.remove_edge(edge[0], edge[1])

            cur_graph_sams = set(graph.nodes)
            missing_sams = all_sam_names - cur_graph_sams
            for missing_sam in missing_sams:
                graph.add_node(missing_sam)
            glob_var.views.update({view: _View(graph=graph, name=view)})
            glob_var.gmm_params.update({view: {'mean': means, 'cov': covs, 'percentile': percentile}})
        return glob_var


    def _create_seeds_from_solution(self, glob_var, init_modules):
        for mod_name, sam_ids in init_modules.items():
            views = glob_var.views
            sam_dict = {}
            for sam_id in sam_ids:
                sam_dict[sam_id] = glob_var.samples[sam_id]
            mod_weight = 0
            for view in views.values():
                mod_weight += view.graph.subgraph(list(sam_dict.keys())).size('weight')
            _Module(glob_var=glob_var, samples=sam_dict, views=views, weight=mod_weight)
        return glob_var


    def _get_seeds(self, glob_var, num_of_seeds=3, num_of_samples_in_seed=10, random_state: int = None):
        """
        Create seed modules.
        """
        lst = list(glob_var.views.items())
        lst.sort(key=lambda x: x[0])
        views_list = [view for name, view in lst]
        sam_list = list(glob_var.samples.keys())
        adj = np.zeros((len(sam_list), len(sam_list)))
        for name, view in lst:
            adj += nx.adjacency_matrix(view.graph.subgraph(sam_list), nodelist=sam_list)
        adj = pd.DataFrame(adj, index=sam_list, columns=sam_list)
        joined_subgraph = nx.from_pandas_adjacency(adj)
        view_graphs = [joined_subgraph]

        for i in range(num_of_seeds):

            view_graph = view_graphs[0]
            cur_nodes = list(sorted(view_graph.nodes()))
            adj = list(view_graph.adjacency())

            if len(cur_nodes) == 0:
                break

            rand_sam_index = np.random.default_rng(random_state + i).integers(0, max([len(cur_nodes) - 1, 1]))
            rand_sam_name = cur_nodes[rand_sam_index]
            rand_sam_in_adj = [sam[0] for sam in adj].index(rand_sam_name)
            neighbors = [(key, adj[rand_sam_in_adj][1][key]['weight']) for key in adj[rand_sam_in_adj][1]]
            neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:(num_of_samples_in_seed - 1)]
            nodes = {rand_sam_name: glob_var.samples[rand_sam_name]}
            for nei in neighbors:
                if nei[1] > 0 and nei[0] != rand_sam_name:
                    nodes.update({nei[0]: glob_var.samples[nei[0]]})
            mod_weight = view_graph.subgraph(list(nodes.keys())).size('weight')
            if mod_weight > 0 and len(nodes) > 1 and len(nodes) >= glob_var.min_mod_size:
                _Module(glob_var=glob_var, samples=nodes, views=[view for view in views_list], weight=mod_weight)
                for k in range(len(view_graphs)):
                    view_graph = view_graphs[k]
                    remaining_nodes = list(sorted(set(cur_nodes) - set(nodes.keys())))
                    view_graphs[k] = view_graph.subgraph(remaining_nodes)
        return glob_var


    def _build_a_graph_similarity(self, distances):
        g = nx.from_numpy_array(distances.values)
        mapping = {i: j for i, j in enumerate(distances.columns)}
        nx.relabel_nodes(g, mapping, False)
        return g, [], [], 0


    def _get_next_step(self, mod, glob_var):
        """
        this function decided what is the next action that will be executed.
        """
        max_res = (-1, (-float("inf"), None))
        for func_i in range(len(self.functions)):
            if func_i <= 9:  # only one module needed
                tmp = self.functions[func_i](mod, glob_var)
                if tmp[0] > max_res[1][0]:
                    max_res = (func_i, tmp)
            else:
                for mod2 in glob_var.modules.values():
                    if mod2 == mod:
                        continue
                    tmp = self.functions[func_i](mod, mod2, glob_var)
                    if tmp[0] > max_res[1][0]:
                        max_res = (func_i, tmp)
        return max_res


    def _exec_next_step(self, mod, max_res, glob_var):
        """
        this function actually performs an action, given that the
        algorithm already decided what the next action will be.
        """
        if max_res[1][0] == -float("inf") or max_res[1][0] < 0:
            return glob_var
        func_i = max_res[0]
        glob_var.actions[func_i] += 1
        if func_i == 0:  # add
            for sample in max_res[1][1]:
                mod.add_sample(sample)
        elif func_i == 1:  # remove
            mod.remove_sample(max_res[1][1])
            if len(mod.get_samples()) <= 1:
                glob_var = glob_var.kill_module(mod)
        elif func_i == 2:  # add view
            mod.add_view(max_res[1][1], glob_var)
        elif func_i == 3:  # remove view
            mod.remove_view(max_res[1][1], glob_var)
        elif func_i == 4:  # split
            glob_var = mod.split_module(max_res[1][1][1], glob_var)
        elif func_i == 5:  # split and add view
            glob_var = mod.split_and_add_view(view=max_res[1][1][0], sub_nodes=max_res[1][1][1], glob_var=glob_var)
        elif func_i == 6:  # split and remove view
            glob_var = mod.split_and_remove_view(view=max_res[1][1][0], sub_nodes=max_res[1][1][1], glob_var=glob_var)
        elif func_i == 7:  # create new module
            new_mod = _Module(glob_var)
            new_mod.add_view(max_res[1][1][1], glob_var)
            for sam in max_res[1][1][0]:
                new_mod.add_sample(glob_var.samples[sam])
        elif func_i == 8:  # transfer
            sams = [(sam, mod2) for sam, weight, mod2 in max_res[1][1]]
            for sam, mod2 in sams:
                _switch_2_samples(glob_var.samples[sam], mod, mod2, glob_var)
        elif func_i == 9:  # spread module
            mod.spread_module(max_res[1][1], glob_var)
        elif func_i == 10:  # merge
            glob_var = mod.merge_with_module(max_res[1], glob_var)
        return glob_var


    def _is_mod_significant(self, mod, glob_var, percentile=95, iterations=500, random_state: int = None):
        """
        Assess the statisitcal significance of a module by sampling modules or similar size.
        """
        draws = [0 for i in range(iterations)]
        mod_size = len(mod.get_samples())
        if mod_size <= 1:
            return False
        for i in range(iterations):
            samps = np.random.default_rng(random_state + i).choice(list(glob_var.samples.keys()),
                                                                        mod_size).tolist()
            lst = list(mod.get_views().items())
            lst.sort(key=lambda x: x[0])
            for name, view in lst:
                draws[i] += view.graph.subgraph(samps).size('weight')
        num_to_beat = np.percentile(draws, percentile)
        return mod.get_weight() > num_to_beat


    @staticmethod
    def _post_processing(glob_var):
        labels = [[sample, mod_id] for mod_id, module in glob_var.modules.items() for sample in module.samples]
        labels = pd.DataFrame(labels)
        labels = labels.set_index(0)
        sams_without_mods = pd.DataFrame(None, index=glob_var.samples.keys())
        labels = pd.concat([labels, sams_without_mods.loc[sams_without_mods.index.difference(labels.index)]])
        view_graphs = [pd.DataFrame(nx.to_numpy_array(view.graph)) for view in glob_var.views.values()]
        mod_views = {mod_name: list(module.get_views().keys()) for mod_name, module in glob_var.modules.items()}
        return labels, view_graphs, mod_views


    # def _get_em_graph_per_view(self, Xs: list, predictions=False):
    #     """gets raw data and return a list of similarity matrices"""
    #
    #     sim_data = [1 - X.T.corr() for X in Xs]
    #     all_views_ret = []
    #     for i, cur_sim in enumerate(sim_data):
    #         if predictions:
    #             num_clusters = self.n_clusters_[i]
    #         else:
    #             n_clusters = list(range(2, 11))
    #             scores = [silhouette_score(cur_sim,
    #                                        SpectralClustering(n_clusters=k, affinity='precomputed', n_jobs = -1,
    #                                                           random_state= self.random_state).fit_predict(cur_sim)) \
    #                       for k in n_clusters]
    #             num_clusters = n_clusters[np.argmax(scores)]
    #
    #         if predictions:
    #             em_ret = self.all_ems_[i]
    #         else:
    #             chosen_sims_mat = cur_sim.sample(frac=1., random_state=42)
    #             chosen_sims_mat = chosen_sims_mat[chosen_sims_mat.index]
    #             chosen_sims = chosen_sims_mat.values[np.triu_indices_from(chosen_sims_mat, k=1)]
    #             em_ret = GaussianMixture(n_components=2, n_init=20, max_iter=int(1e5),
    #                                      random_state= self.random_state).fit(pd.DataFrame(chosen_sims))
    #
    #         # calculate probabilities
    #         sigma = [np.sqrt(np.trace(em_ret.covariances_[i]) / 2) for i in range(2)]
    #         prob1 = np.log(em_ret.weights_[0]) + norm.logpdf(cur_sim, loc=em_ret.means_[0], scale=sigma[0])
    #         prob2 = np.log(em_ret.weights_[1]) + norm.logpdf(cur_sim, loc=em_ret.means_[1], scale=sigma[1])
    #         prob = prob1 - prob2 if em_ret.means_[0] < em_ret.means_[1] else prob2 - prob1
    #         shift_by = np.quantile(prob[np.triu_indices_from(prob, k=1)], 1 - 1 / num_clusters)
    #         prob = prob - shift_by
    #         np.fill_diagonal(prob, 0)
    #         prob = pd.DataFrame(prob, index = Xs[i].index.astype(str), columns=Xs[i].index.astype(str))
    #         all_views_ret.append({"prob": prob, "all_views_ems": em_ret, "num_clusters": num_clusters})
    #     return all_views_ret


    # def _monet_ret_to_module_membership(self, view_graphs=None):
    #     if view_graphs is None:
    #         view_graphs = self.view_graphs_
    #     mod_views = self.mod_views_
    #     mod_names = list(self.glob_var_.modules.keys())
    #     samples = self.labels_.index
    #     all_module_membership = []
    #     for mod_name in mod_names:
    #         cur_mod_views = mod_views[mod_name]
    #         cur_mod_samples = samples[self.labels_ == mod_name] + 'fit'
    #         cur_module_membership = sum([view_graphs[int(i)][cur_mod_samples].sum(axis=0) for i in cur_mod_views])
    #         all_module_membership.append(cur_module_membership.tolist())
    #     all_module_membership = pd.DataFrame(all_module_membership, index=mod_names)
    #     all_module_membership = all_module_membership.T.idxmax(1)
    #     all_module_membership = all_module_membership[all_module_membership.index.difference(samples.astype(int))]
    #     return all_module_membership
