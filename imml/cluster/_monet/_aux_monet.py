# License: BSD-3-Clause

import copy
import heapq
import networkx as nx
import numpy as np


class _Globals:
    """
    The class contains the state of a MONET run - all samples, modules, view graphs, and configuration.
    """

    def __init__(self, len):
        self.samples = {}
        self.modules = {}
        self.converged_modules = {}
        self.active_modules = {}
        self.views = {}
        self.actions = [0 for i in range(len)]
        self.index = 0
        self.min_mod_size = 1
        self.max_sams_per_action = 10
        self.gmm_params = {}
        return

    def kill_module(self, mod):
        sams = mod.get_samples().copy().items()
        for name, sam in sams:
            mod.remove_sample(sam)
        if isinstance(mod, int):
            del self.modules[mod]
        elif isinstance(mod, _Module):
            for name, module in self.modules.items():
                if module == mod:
                    break
            del self.modules[name]
        else:
            raise Exception('unknown module id')
        return self

    def add_module(self, mod):
        self.modules.update({self.index : mod})
        self.index += 1
        return


class _Module:
    """
    Represents a single module (set of samples and views).
    Contains functions to manipulate modules which are used by MONET actions
    (e.g. split module).
    """

    def __init__(self, glob_var, samples={}, views={}, weight=0):
        self.samples = {}
        self.views = {}
        self.weight = 0
        for name, sam in samples.items():
            self.add_sample(sam)
        for view in views:
            self.add_view(view, glob_var)
        if weight:
            if abs(self.weight - weight) > 0.0001:
                # This error should never occur!
                print("self.weight != weight")
                exit(1)
        glob_var.add_module(self)
        return

    def is_sample_in_module(self, sam):
        return sam in self.samples

    def get_samples(self):
        return self.samples

    def get_size(self):
        return len(self.samples)

    def add_sample(self, sam):
        self.samples.update({sam.get_name(): sam})
        sam_lst = self.get_samples_names_as_list()
        for view in self.views.values():
            edges = view.graph.subgraph(sam_lst).edges(sam.get_name(), data=True)
            for e in edges:
                self.weight += e[2]['weight']
        sam.set_module(self)
        return self.weight

    def remove_sample(self, sam):
        if isinstance(sam, (str, int)):
            sam = self.get_samples()[sam]
        sam.remove_module(self)
        sam_lst = self.get_samples_names_as_list()
        for view in self.views.values():
            edges = view.graph.subgraph(sam_lst).edges(sam.get_name(), data=True)
            for e in edges:
                self.weight -= e[2]['weight']
        try:
            self.samples.pop(sam.get_name())
        except:
            # This error should never occur!
            raise Exception("The clusters could not be found.")
        return self.weight

    def get_views(self):
        return self.views

    def add_weight(self, weight):
        self.weight += weight
        return self.weight

    def dec_weight(self, weight):
        self.weight -= weight
        return self.weight

    def get_weight(self):
        return self.weight

    # returns but does not change the object
    def calc_module_weight(self):
        weight = 0
        for view in self.views.values():
            g1 = view.graph.subgraph(list(self.samples.keys()))
            weight += g1.size('weight')
        return weight

    def get_samples_names_as_list(self):
        return [name for name, node in self.get_samples().items()]

    def add_view(self, view, glob_var=None):
        if isinstance(view, str):
            view_name = view
            view = glob_var.views[view]
        else:
            for name, obj in glob_var.views.items():
                if obj == view:
                    view_name = name
                    break
        if view_name not in self.views:
            self.weight += view.graph.subgraph(self.get_samples_names_as_list()).size("weight")
            self.views[view_name] = view
        return self.weight

    def remove_view(self, view, glob_var=None):
        if isinstance(view, str):
            view_name = view
            view = self.views[view_name]
        else:
            for name, obj in glob_var.views.items():
                if obj == view:
                    view_name = name
                    break
        if view_name in self.views:
            self.weight -= view.graph.subgraph(self.get_samples_names_as_list()).size("weight")
            del self.views[view_name]
        return self.weight

    def merge_with_module_union(self, mod1, glob_var):
        for view in mod1.get_views().values():
            if view not in self.views:
                self.add_view(view, glob_var)
        for name in mod1.get_samples_names_as_list():
            mod1.remove_sample(glob_var.samples[name])
            self.add_sample(glob_var.samples[name])
        glob_var.kill_module(mod1)
        return glob_var

    def eat_module(self, mod, glob_var):
        lst = [(name, sam) for name, sam in mod.get_samples().items()]
        for (name, sam) in lst:
            mod.remove_sample(sam)
            self.add_sample(sam)
        glob_var.kill_module(mod)
        return glob_var

    def merge_me_into(self, mod, glob_var):
        mods_sams = [(name, sam) for name, sam in mod.get_samples().items()]
        selfs_views_list = list(self.get_views().items())
        # selfs_views_list.sort(key=lambda x: x[0])
        for name, sam in mods_sams:
            mod.remove_sample(sam)
            self.add_sample(sam)
        for name, sam in selfs_views_list:
            self.remove_view(sam, glob_var)
        for name, sam in mod.get_views().items():
            self.add_view(sam, glob_var)
        glob_var.kill_module(mod)
        return glob_var

    def merge_to_intersection_views(self, mod, glob_var):
        lst = [(name, sam) for name, sam in mod.get_samples().items()]
        for name, sam in lst:
            mod.remove_sample(sam)
            self.add_sample(sam)
        views_lst = list(self.get_views().items())
        views_lst.sort(key=lambda x: x[0])
        for name, view in views_lst:
            if view not in mod.get_views().values():
                self.remove_view(view, glob_var)
        glob_var.kill_module(mod)
        return glob_var

    def split_module(self, sub_nodes, glob_var):
        sam_list = self.get_samples_names_as_list()
        left_out = set(sub_nodes).symmetric_difference(set(sam_list))
        new_mod = _Module(glob_var)
        for view in self.get_views():
            new_mod.add_view(view, glob_var)
        for sam in left_out:
            self.remove_sample(glob_var.samples[sam])
            new_mod.add_sample(glob_var.samples[sam])
        return glob_var

    def split_and_add_view(self, view, sub_nodes, glob_var):
        sams = self.get_samples_names_as_list()
        left_out = set(sams).symmetric_difference(set(sub_nodes))
        if left_out:  # not empty
            new_mod = _Module(glob_var=glob_var, views=self.get_views())
            for sam in left_out:
                self.remove_sample(sam)
                new_mod.add_sample(glob_var.samples[sam])
        self.add_view(view, glob_var)
        return glob_var

    def split_and_remove_view(self, view, sub_nodes, glob_var):
        sams = self.get_samples_names_as_list()
        sams_in_new_module = set(sub_nodes)
        sams_remain_in_this_module = set(sams).symmetric_difference(set(sams_in_new_module))
        if sams_in_new_module and (sams_remain_in_this_module or len(self.views) > 1):  # not empty
            new_mod = _Module(glob_var=glob_var, views=[view])
            for sam in sams_in_new_module:
                self.remove_sample(sam)
                new_mod.add_sample(glob_var.samples[sam])
        return glob_var

    def merge_with_module(self, params, glob_var):
        if params[1][0] == 0:
            return self.merge_with_module_union(params[1][1][1], glob_var)
        if params[1][0] == 1:
            return self.eat_module(params[1][1][1], glob_var)
        if params[1][0] == 2:
            return self.merge_me_into(params[1][1][1], glob_var)
        if params[1][0] == 3:
            return self.merge_to_intersection_views(params[1][1][1], glob_var)

    def spread_module(self, params, glob_var):
        for sam, new_mod in params.items():
            self.remove_sample(sam)
            if new_mod is not None:
                new_mod.add_sample(sam)
        glob_var.kill_module(self)


class _Sample:
    """
    Represents a single sample.
    """


    def __init__(self, name):
        self.module = None
        self.name = name

    def set_module(self, module):
        if self.is_in_module():
            raise NameError(f'trying to add sample {self.name} to 2 modules', self)
        self.module = module
        return

    def remove_module(self, module):
        self.module = None
        return module

    def get_module(self):
        return self.module

    def is_in_module(self):
        return self.module is not None

    def __str__(self):
        return self.name

    def get_name(self):
        return self.name


class _View:
    """
    Represents a single view, and contains the view's graph.
    """

    def __init__(self, graph=None, name=""):
        self.graph = graph
        self.name = name
        return

    def set_graph(self, g):
        self.graph = g
        return

    def get_name(self):
        return self.name

    def __repr__(self):
        return self.name


def _best_samples_to_add(module, glob_var):
    """"returns the best neighbors that max the immediate score and the score it would provide
    in case there is no such node will return None and -inf"""
    heap = []
    max_sams_to_add = glob_var.max_samps_per_action
    start_weight = module.get_weight()
    best_sample = (-float("inf"), None)

    for name, sam in glob_var.samples.items():
        if sam.is_in_module():
            continue

        tmp_weight = module.add_sample(sam)
        module.remove_sample(sam)

        if tmp_weight < start_weight:
            continue
        if tmp_weight > best_sample[0]:
            best_sample = tmp_weight, sam
        if len(heap) < max_sams_to_add:
            heapq.heappush(heap, (tmp_weight, name))
        elif tmp_weight > heap[0][0]:
            heapq.heappushpop(heap, (tmp_weight, name))

    for weight, sample_name in heap:
        module.add_sample(glob_var.samples[sample_name])
    total_addition = module.get_weight()
    for weight, sample_name in heap:
        module.remove_sample(glob_var.samples[sample_name])

    if best_sample[0] > total_addition:
        maximal_addition = best_sample[0]
        to_add = [best_sample[1]]
    else:
        maximal_addition = total_addition
        to_add = [glob_var.samples[sample_name] for weight, sample_name in heap]
    if maximal_addition > start_weight and len(to_add) > 0:
        return maximal_addition - start_weight, to_add
    return -float("inf"), None


def _which_sample_to_remove(module, glob_var):
    """returns the node that is most beneficial to remove and
    the score it would provide in case there is no such node will return None and -inf"""
    if len(module.get_samples()) == glob_var.min_mod_size:
        return -float("inf"), None
    cur_max = module.get_weight()
    start_weight = module.get_weight()
    sam_to_del = None
    for name, sam in copy.deepcopy(module.get_samples()).items():
        tmp_weight = module.remove_sample(sam)
        if tmp_weight > cur_max:
            cur_max = tmp_weight
            sam_to_del = sam
        module.add_sample(sam)
    if sam_to_del:
        return cur_max - start_weight, sam_to_del
    return -float("inf"), None


def _top_samples_to_switch(mod, glob_var):
    max_sams_to_switch = glob_var.max_samps_per_action
    res = []
    init_weight = sum([tmp_mod.get_weight() for mod_name, tmp_mod in glob_var.modules.items()])
    for mod2_name, mod2 in glob_var.modules.items():
        if mod == mod2:
            continue
        weight, samples_list = _best_samples_to_switch(mod, mod2, glob_var)
        res += samples_list
        cur_num = (len(mod.get_samples_names_as_list()))
    if len(res) == 0:
        return -float("inf"), None
    res = sorted(res, key=lambda x: x[1], reverse=True)
    used = set()
    used_indices = set()
    for i, (sam, score, mod2) in enumerate(res):
        if sam in used:
            continue
        sam_mod = glob_var.samples[sam].module
        if len(sam_mod.get_samples()) == glob_var.min_mod_size:
            continue
        if len(used) == max_sams_to_switch:
            break
        _switch_2_samples(sam, mod, mod2, glob_var)
        used.add(sam)
        used_indices.add(i)
    curr_weight = sum([tmp_mod.get_weight() for mod_name, tmp_mod in glob_var.modules.items()])
    ret = []
    for i, (sam, score, mod2) in enumerate(res):
        if sam in used and i in used_indices:
            used.remove(sam)
            _switch_2_samples(sam, mod, mod2, glob_var)
            ret += [(sam, score, mod2)]
    res = ret
    if len(res) == 0:
        return -float("inf"), None
    if (curr_weight - init_weight) > res[0][1]:
        return curr_weight - init_weight, res
    return res[0][1], [res[0]]


def _best_samples_to_switch(mod1, mod2, glob_var):
    mod1_names = mod1.get_samples_names_as_list()
    mod2_names = mod2.get_samples_names_as_list()
    samples_dict = {node: {"in_mod": 0, "diff_mod": 0} for node in (mod1_names + mod2_names)}
    start_weight = mod1.get_weight() + mod2.get_weight()
    for view_name, view in mod1.get_views().items():
        for node_name, degree in view.graph.subgraph(mod1_names).degree(weight='weight'):
            samples_dict[node_name]["in_mod"] += degree
        for node_name in mod2_names:
            samples_dict[node_name]["diff_mod"] += \
                view.graph.subgraph(mod1_names + [node_name]).degree(node_name, weight="weight")
    for view_name, view in mod2.get_views().items():
        for node_name, degree in view.graph.subgraph(mod2_names).degree(weight='weight'):
            samples_dict[node_name]["in_mod"] += degree
        for node_name in mod1_names:
            samples_dict[node_name]["diff_mod"] += \
                view.graph.subgraph(mod2_names + [node_name]).degree(node_name, weight="weight")
    samples_list = [(node, dict["diff_mod"] - dict["in_mod"], mod2) for node, dict in samples_dict.items()]
    samples_list = sorted(samples_list, key=lambda x: x[1], reverse=True)
    num_switches = 0

    for i in range(min(len(samples_list), glob_var.max_samps_per_action)):
        if samples_list[i][1] <= 0:
            if i == 0:
                return -float("inf"), []
            break
        num_switches += 1
        _switch_2_samples(samples_list[i][0], mod1, mod2, glob_var)
    samples_list = samples_list[:num_switches]
    switch_all_weight = mod1.get_weight() + mod2.get_weight() - start_weight
    for sam, diff, fake_mod in samples_list:
        _switch_2_samples(sam, mod1, fake_mod, glob_var)

    if switch_all_weight < samples_list[0][1]:
        samples_list = [samples_list[0]]
        switch_all_weight = samples_list[0][1]
    if switch_all_weight > 0:
        return switch_all_weight, samples_list
    return -float("inf"), []


def _switch_2_samples(sam, mod1, mod2, glob_var):
    if isinstance(sam, (str, int)):
        sam = glob_var.samples[sam]
    if sam.get_module() == mod1:
        mod1.remove_sample(sam)
        mod2.add_sample(sam)
    elif sam.get_module() == mod2:
        mod2.remove_sample(sam)
        mod1.add_sample(sam)
    else:
        # This should never occur!
        raise Exception("The clusters could not be found.")
    return


def _which_view_to_add_to_module(mod, glob_var):
    """get a module and see what happens when you add different views"""
    cur_max = mod.get_weight(), None
    start_weight = mod.get_weight()
    for view in glob_var.views:
        if view in mod.get_views():
            continue
        mod.add_view(view, glob_var)
        tmp = mod.get_weight()
        if tmp > cur_max[0]:
            cur_max = tmp, view
        mod.remove_view(view, glob_var)
    if cur_max[1]:
        return cur_max[0] - start_weight, cur_max[1]
    return -float("inf"), None


def _which_view_to_remove_from_module(mod, glob_var):
    cur_max = mod.get_weight(), None
    start_weight = mod.get_weight()
    if len(mod.get_views()) <= 1:
        return -float("inf"), None
    for view in copy.deepcopy(mod.get_views()):
        mod.remove_view(view, glob_var)
        tmp = mod.get_weight()
        if tmp > cur_max[0]:
            cur_max = tmp, view
        mod.add_view(view, glob_var)
    if cur_max[1]:
        return cur_max[0] - start_weight, cur_max[1]
    return -float("inf"), None


def _weight_of_merged_modules(mod1, mod2, glob_var):
    start_weight = mod1.get_weight() + mod2.get_weight()
    samples_list = mod1.get_samples_names_as_list() + mod2.get_samples_names_as_list()
    views_lists = [set(mod1.get_views().values()) | set(mod2.get_views().values()), set(mod1.get_views().values()),
                   set(mod2.get_views().values()),
                   set(mod1.get_views().values()).intersection(set(mod2.get_views().values()))]
    weights = [0 for i in views_lists]
    for i in range(len(weights)):
        for view in views_lists[i]:
            weights[i] += view.graph.subgraph(samples_list).size('weight')
    if max(weights) < start_weight:
        return -float("inf"), None
    return max(weights) - start_weight, (weights.index(max(weights)), (mod1, mod2))


def _score_of_split_module(mod, glob_var):
    start_weight = mod.get_weight()
    adj = np.zeros((len(mod.get_samples()), len(mod.get_samples())))
    sam_list = mod.get_samples_names_as_list()
    if len(sam_list) == 0:
        raise Exception("no samples in split module {}.".format(mod))
    lst = list(mod.get_views().items())
    lst.sort(key=lambda x: x[0])
    for name, view in lst:
        adj += nx.adjacency_matrix(view.graph.subgraph(sam_list), nodelist=sam_list)
    joined_subgraph = nx.from_numpy_array(adj)
    mapping = {i: j for i, j in enumerate(sam_list)}
    joined_subgraph = nx.relabel_nodes(joined_subgraph, mapping)
    heavy_subweight, heavy_subnodes = _find_heaviest_subgraph(joined_subgraph, weight=mod.get_weight(),
                                                              min_size=glob_var.min_mod_size,
                                                              max_size=len(sam_list) - glob_var.min_mod_size)
    if heavy_subnodes is None:
        return -float("inf"), None
    left_out = set(heavy_subnodes).symmetric_difference(set(sam_list))
    if heavy_subnodes is None or len(left_out) <= 1 or len(heavy_subnodes) <= 1:
        return -float("inf"), None
    other_weight = joined_subgraph.subgraph(left_out).size('weight')
    heavy_weight = joined_subgraph.subgraph(heavy_subnodes).size('weight')
    end_weight = heavy_weight + other_weight
    if start_weight > end_weight:
        return -float("inf"), None
    else:
        return end_weight - start_weight, (mod, heavy_subnodes)


def _find_heaviest_subgraph(orig_g, weight=0, min_size=1, max_size=None):
    if max_size is None:
        max_size = len(orig_g.nodes())
    if not weight:
        weight = orig_g.size('weight')
    max_subgraph = -float("inf"), None
    g = orig_g.copy()
    graph_size = g.number_of_nodes()
    nodes_degrees = {node_name: node_degree for node_name, node_degree in g.degree(weight='weight')}
    while graph_size > 1:
        lightest_node = min(list(nodes_degrees.items()), key=lambda x: x[1])
        nodes_degrees.pop(lightest_node[0])
        for u, v in g.edges(lightest_node[0]):
            if u == lightest_node[0]:
                nodes_degrees[v] -= g.edges[u, v]['weight']
        g.remove_node(lightest_node[0])
        graph_size -= 1
        weight -= lightest_node[1]
        cur_graph_size = len(g.nodes())
        if weight > max_subgraph[0] and cur_graph_size >= min_size and cur_graph_size <= max_size:
            max_subgraph = weight, list(g.nodes())
    return max_subgraph


def _weight_of_split_and_add_view(mod, glob_var):
    pot_views = set(glob_var.views.values()).symmetric_difference(set(mod.get_views().values()))
    sams = mod.get_samples_names_as_list()
    start_weight = mod.get_weight()
    max_improvement = -float("inf"), None
    for view in pot_views:
        tmp_graph = view.graph.subgraph(sams)
        tmp_weight = tmp_graph.size('weight')
        weight, sub_nodes = _find_heaviest_subgraph(tmp_graph, tmp_weight,
                                                    min_size=glob_var.min_mod_size,
                                                    max_size=len(sams) - glob_var.min_mod_size)
        if sub_nodes is None:
            continue
        weight2 = 0
        for sub_view in mod.get_views().values():
            weight += sub_view.graph.subgraph(sub_nodes).size('weight')
            weight2 += sub_view.graph.subgraph(set(sams).symmetric_difference(set(sub_nodes))).size('weight')
        if weight + weight2 - start_weight > max_improvement[0]:
            max_improvement = (weight + weight2 - start_weight), (view, sub_nodes)
    if max_improvement[0] > 0:
        return max_improvement
    else:
        return -float("inf"), None


def _weight_of_split_and_remove_view(mod, glob_var):
    # With 1 view this is equivalent to module splitting.
    pot_views = set(mod.get_views().values())
    sams = mod.get_samples_names_as_list()
    start_weight = mod.get_weight()
    max_improvement = -float("inf"), None
    for view in pot_views:
        tmp_graph = view.graph.subgraph(sams)
        tmp_weight = tmp_graph.size('weight')
        weight, sub_nodes = _find_heaviest_subgraph(tmp_graph, tmp_weight,
                                                    min_size=glob_var.min_mod_size,
                                                    max_size=len(sams) - glob_var.min_mod_size)
        if sub_nodes is None:
            continue
        weight2 = 0
        # for sub_view in pot_views.symmetric_difference(set([view])):
        for sub_view in pot_views:
            weight2 += sub_view.graph.subgraph(set(sams).symmetric_difference(set(sub_nodes))).size('weight')
        if weight + weight2 - start_weight > max_improvement[0]:
            max_improvement = (weight + weight2 - start_weight), (view, sub_nodes)
    if max_improvement[0] > 0:
        return max_improvement
    else:
        return -float("inf"), None


def _weight_of_new_module(mod, glob_var):
    free_samples = []
    max_weight = -float("inf")
    max_view = None
    max_samples = None
    for sam_name, sam in glob_var.samples.items():
        if not sam.is_in_module():
            free_samples += [sam_name]
    for view_name, view in glob_var.views.items():
        tmp_weight, tmp_nodes = _find_heaviest_subgraph(view.graph.subgraph(free_samples),
                                                        min_size=glob_var.min_mod_size)
        if tmp_nodes is None:
            continue
        if tmp_weight > max_weight:
            max_weight = tmp_weight
            max_view = view
            max_samples = tmp_nodes
    if max_weight > 0 and len(max_samples) >= 2:
        return max_weight, (max_samples, max_view)
    else:
        return -float("inf"), None


def _weight_of_spreading_module(mod, glob_var):
    if len(mod.samples) > 2 * glob_var.min_mod_size:
        return -float("inf"), None
    mod_weight = mod.weight
    new_sam_mods = {}
    total_weight = sum([cur_mod.weight for cur_mod in glob_var.modules.values()])
    orig_mod_sams = []
    for sam in set(mod.samples.values()).copy():
        orig_mod_sams.append(sam)
        sam_best = 0, None
        mod.remove_sample(sam)
        for cur_mod_name, cur_mod in glob_var.modules.items():
            if cur_mod == mod:
                continue
            before_mod_weight = cur_mod.weight
            cur_mod.add_sample(sam)
            after_mod_weight = cur_mod.weight
            cur_mod.remove_sample(sam)
            weight_diff = after_mod_weight - before_mod_weight
            if weight_diff > sam_best[0]:
                sam_best = weight_diff, cur_mod
        mod.add_sample(sam)
        new_sam_mods[sam] = sam_best[1]

    for sam in orig_mod_sams:
        sam_new_mod = new_sam_mods[sam]
        mod.remove_sample(sam)
        if sam_new_mod is not None:
            sam_new_mod.add_sample(sam)
    new_total_weight = sum([cur_mod.weight for cur_mod in glob_var.modules.values()])
    weight_diff = new_total_weight - total_weight

    for sam in orig_mod_sams:
        sam_new_mod = new_sam_mods[sam]
        if sam_new_mod is not None:
            sam_new_mod.remove_sample(sam)
        mod.add_sample(sam)

    if weight_diff > 0:
        return weight_diff, new_sam_mods
    else:
        return -float("inf"), None



