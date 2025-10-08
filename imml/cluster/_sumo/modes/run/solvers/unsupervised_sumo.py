# License: BSD-3-Clause

import numpy as np

from ....._sumo.modes.run.solver import SumoSolver, SumoNMFResults
from ....._sumo.modes.run.utils import svd_h_init, svd_si_init
from ....._sumo.network import MultiplexNet


class UnsupervisedSumoNMF(SumoSolver):
    """ Unsupervised SUMO solver (A(i)=HS(i)H^T formulation) """

    def __init__(self, graph: MultiplexNet, nbins: int, bin_size: int = None, random_state: int = None,
                 verbose: bool = False):

        super(UnsupervisedSumoNMF, self).__init__(graph=graph, nbins=nbins, bin_size=bin_size, verbose=verbose)

        # create average adjacency matrix
        self.avg_adj = self.calculate_avg_adjacency()

        # create sample bins
        self.bins = self.create_sample_bins(random_state=random_state)

        # layer impact balancing parameters
        self.lambdas = [(1. / samples.shape[0]) ** 2 for samples in self.graph.samples]
        self.random_state = random_state

    def factorize(self, sparsity_penalty: float, k: int, max_iter: int = 500, tol: float = 1e-5, calc_cost: int = 1,
                  h_init: int = None, bin_id: int = None) -> SumoNMFResults:
        """ Run tri-factorization

        Args:
            sparsity_penalty (float): 'η' value, corresponding to sparsity penalty for H
            k (int): expected number of clusters
            max_iter (int): maximum number of iterations
            tol (float): if objective cost function value fluctuation is smaller than 'stop_val', \
                stop iterations before reaching max_iter
            calc_cost (int): number of steps between every calculation of objective cost function
            h_init (int): index of adjacency matrix to use for H matrix initialization or None for initialization \
                using average adjacency
            bin_id (int): id of sample bin created in SumoNMF constructor (default of None, means clustering \
                all samples instead of samples in given bin)

        Returns:
            SumoNMFResults object (with result feature matrix / soft cluster indicator matrix (H array),
            and the list of result S matrices for each graph layer)
        """
        eps = np.spacing(1)  # epsilon

        if k > self.graph.nodes:
            raise ValueError("Expected number of clusters greater than number of nodes in graph - expected k << nodes!")

        if bin_id is None:
            sample_ids = np.arange(self.graph.nodes)
        else:
            sample_ids = self.bins[bin_id]

        # filter missing samples
        for layer in self.graph.adj_matrices:
            layer[np.isnan(layer)] = 0.

        a = np.array([self.graph.adj_matrices[i][sample_ids, sample_ids[:, None]] for i in range(self.graph.layers)])
        w = np.array([self.graph.w[i][sample_ids, sample_ids[:, None]] for i in range(self.graph.layers)])
        wa = np.array([w[i] * a[i] for i in range(self.graph.layers)])

        # randomize S matrices for each layer
        s = np.array([svd_si_init(self.graph.adj_matrices[i], k, self.random_state) for i in range(self.graph.layers)])

        # randomize feature matrix / soft cluster indicator matrix
        h = svd_h_init(self.graph.adj_matrices[h_init] if h_init is not None else self.avg_adj, k)[sample_ids, :]

        # objective function
        objval = np.zeros((max_iter + 1, self.graph.layers + 2))
        step = 0
        step_recorded = 0
        stop = np.inf
        before_val = np.inf
        best_cost = np.inf
        best_result = (0, h, s)  # step, H, S(i)

        while step <= max_iter and stop > tol:

            # update s matrices
            for i in range(len(s)):
                num = h.T @ wa[i] @ h
                den = h.T @ (w[i] * (h @ s[i] @ h.T)) @ h
                s[i] = s[i] * ((num + eps) / (den + eps))

            # update h
            num, den = np.zeros((sample_ids.size, k)), np.zeros((sample_ids.size, k))
            for i in range(len(s)):
                num = num + ((self.lambdas[i] * wa[i]) @ h @ s[i])
                den = den + (self.lambdas[i] * (w[i] * (h @ s[i] @ h.T)) @ h @ s[i] + 0.5 * sparsity_penalty * h)
            h = h * ((num + eps) / (den + eps))

            if step % calc_cost == 0 or step == max_iter:

                for i in range(len(s)):
                    objval[step_recorded, i] = self.lambdas[i] * np.sum(
                        w[i] * (np.nansum(np.dstack((a[i], -(h @ (s[i] @ h.T)))), 2)) ** 2)
                objval[step_recorded, self.graph.layers] = sparsity_penalty * np.sum(h ** 2)
                objval[step_recorded, - 1] = np.sum(objval[step_recorded, :])

                after_val = objval[step_recorded, - 1]
                stop = abs(after_val - before_val) / after_val

                if objval[step_recorded, -1] <= best_cost:
                    best_cost = objval[step_recorded, -1]
                    best_result = (step, h.copy(), s.copy())

                if step == 0:
                    if self.verbose:
                        print("Initial ℒ/Δℒ: {}\t[{} + {}]".format(round(objval[step_recorded, -1], 6),
                                                                          round(np.sum(objval[step_recorded, :-2]), 6),
                                                                          round(objval[step_recorded, -2], 6)))
                else:
                    # objective function value decreases
                    if self.verbose:
                        print("Step({}),\t ℒ: {}\t[{} + {}]".format(step, round(objval[step_recorded, -1], 6),
                                                                           round(np.sum(objval[step_recorded, :-2]), 6),
                                                                           round(objval[step_recorded, -2], 6)))
                before_val = after_val
                step_recorded += 1

            step += 1

        if stop < tol:
            # stop condition was achieved
            if self.verbose:
                print("Stop condition for iterations achieved (|Δℒ| < |{}|).".format(tol))
        else:
            # maximum iterations was reached
            if self.verbose:
                print("Maximum iterations ({}) reached".format(max_iter))

        objval = objval[:step_recorded, :]
        if self.verbose:
            print("#Best achieved results:")

        for i in range(len(best_result[2])):
            if self.verbose:
                print("- Final S({}):\n{}".format(i, best_result[2][i]))
        if self.verbose:
            print("- Final H:\n{}".format(best_result[1]))
            print("- Final ℒ: {} ({}) [step: {}]".format(round(objval[-1, -1], 6),
                                                                round(np.sum(objval[-1, :-2]), 6),
                                                                best_result[0]))

        return SumoNMFResults(graph=self.graph, h=h, s=s, objval=objval, steps=step - 1,
                              sample_ids=sample_ids)
