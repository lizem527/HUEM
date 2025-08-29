import os

from offline.cdf import *
from offline.dis import *
# from offline.new_dis import *
from utils.predicates import *
from utils.workload import *


class TreeBuilder:
    def __init__(self, df, df_sample, config, args, k, output_dir, verbose=True):
        self.df = df
        self.sample = df_sample
        self.cfg = config
        self.args = args
        self.k = k
        self.policy = args.policy.split(",")
        self.dir = {}
        for p in self.policy:
            out = "resources/labels/%s/%s" % (p, output_dir)
            if not os.path.exists(out):
                os.makedirs(out)
            self.dir[p] = out
        self.N = len(df)
        self.min_size = self.N // self.k // 2
        self.method = args.method
        self.verbose = verbose
        self.dir_col = {}
        self.out_dir = "resources/info/%s/%s" % (p, output_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def load_by_path(self, path, use_zorder=False, sample=False, cols='', init=None, workload=None):
        # use_dis
        if self.method == 'd':
            p = DIS(self.df, self.cfg, self.k, 24, workload, self.verbose, init)
        elif self.method == 'c':
            p = CDF(self.df, self.cfg, self.k, 24, None, self.verbose, init)
        label_file = path
        if not '-label' in path and self.method != "qd":
            label_file = path + '-label'
        label_exist = os.path.exists(label_file)

        if not sample and label_exist:
            print("Loading label: %s" % label_file)
            labels = pickle.load(open(label_file, 'rb'))
            p.load_by_labels(labels)
        else:
            p.load_by_path(path)
        # Save partition labels for each row
        # not sample and
        if not sample and not label_exist:
            p.save_labels(label_file)
            print("Saving label %s" % label_file)
        return p

    def load_trees_in_dir(self):
        all_trees = []
        for p in self.policy:
            dir = self.dir[p]
            files = [f for f in listdir(dir) if isfile(join(dir, f))]
            files = sorted(files)
            for file in files:
                if '-label' in file:
                    continue
                # sample?
                tree = DIS(self.sample, self.cfg, [], self.k, None, self.verbose)
                tree.load_by_path(join(dir, file))
                all_trees.append(tree)
        return all_trees

    def compute_optimal_layout(self, queries, path, init):
        if self.method == 'd':
            dl = DIS(self.df, self.cfg, self.k, 24, queries, self.verbose, init)
        elif self.method == 'c':
            dl = CDF(self.df, self.cfg, self.k, 24, queries, self.verbose, init)

        dl.make_partitions()
        dl.save_by_path(path)
        return dl

    def compute_offline_oracle(self, ds, workloads, use_sample=True):
        qfile = self.args.q
        fname = "%s-%s-%d-%s.p" % (
            ds, qfile, self.k, self.method)
        if self.method == 'd':
            dl = DIS(self.df, self.cfg, self.k, 24, workloads, self.verbose)
        elif self.method == 'c':
            dl = CDF(self.df, self.cfg, self.k, 24, workloads, self.verbose)
        dl.make_partitions()
        path = "resources/labels/offline/%s" % fname
        dl.save_by_path(path)
        return dl

    def load_offline_oracle(self, ds, workload, use_sample=False):
        qfile = self.args.q
        if use_sample:
            tree = DIS(self.sample, self.cfg, workload, self.k, None, self.verbose)
        else:
            tree = DIS(self.df, self.cfg, workload, self.k, None, self.verbose)
        path = "resources/labels/offline/%s-%s-%d-qd.p" % (ds, qfile, self.k)

        label_file = path + '-label'
        label_exist = os.path.exists(label_file)
        if not use_sample and label_exist:
            print("Loading label: %s" % label_file)
            labels = pickle.load(open(label_file, 'rb'))
            tree.load_by_labels(labels)
            tree.path = path
        else:
            tree.load_by_path(path)
        # Save partition labels for each row
        if not use_sample and not label_exist:
            tree.save_labels(label_file)
            print("Saving label %s" % label_file)
        return tree

    def get_init_states(self, queries=None):
        init_states = []
        # Oracle: best tree for each workload
        if "oracle" in self.policy:
            if self.args.load:
                init_states = self.load_trees_in_dir()
            else:
                init_states = []
                keys = sorted(queries.keys())
                for i, key in enumerate(keys):
                    path = "%s/%d.p" % (self.dir["oracle"], i)
                    tree = self.compute_optimal_layout(queries[key], path)
                    init_states.append(tree)
            print("Total candidates: %d" % len(init_states))
        elif self.method == "c":
            print("----!!!!!!!!!!  CDF")
            init = CDF(self.df, self.cfg, self.k, 24, None, self.verbose)
            path = "%s/%d.p-label" % (self.dir[self.policy[0]], 0)
            if os.path.exists(path):
                init.path = path
                labels = pickle.load(open("%s" % path, "rb"))
                init.load_by_labels(labels)
            else:
                init.make_partitions()
                print("self.policy:", self.policy)
                for p in self.policy:
                    print("p:", p)
                    path = "%s/%d.p-label" % (self.dir[p], 0)
                    init.save_by_path(path)
            init_states.append(init)
        elif self.method == "d":
            print("----!!!!!!!!!!  DIS")
            init = DIS(self.df, self.cfg, self.k, 24, None, self.verbose)
            path = "%s/%d.p-label" % (self.dir[self.policy[0]], 0)
            if os.path.exists(path):
                init.path = path
                labels = pickle.load(open("%s" % path, "rb"))
                # init.load_by_labels(labels)
                init.load_by_path(path)
            else:
                init.make_partitions()
                print("self.policy:", self.policy)
                for p in self.policy:
                    print("p:", p)
                    path = "%s/%d.p-label" % (self.dir[p], 0)
                    init.save_by_path(path)
            init_states.append(init)
        return init_states
