from utils.tree import *
from utils.workload import *
from utils.reservoir import *


class StateGenerator:
    """Generate new layouts based on sliding window/reservoir sample of queries"""

    def __init__(self, tree_builder, init_states, interval, epsilon=0.08, load=False):
        self.tree_builder = tree_builder
        self.min_size = len(tree_builder.sample) // tree_builder.k // 2
        self.policy = tree_builder.policy
        self.output_dir = tree_builder.dir
        print(self.output_dir)
        self.load = load
        self.reservoir = WeightedReservoir(1000, interval, self.policy)
        self.interval = interval
        self.all_states = list(init_states)
        self.counter = {}
        for p in self.policy:
            self.counter[p] = len(init_states)
        self.cost_profile = []
        self.epsilon = epsilon
        self.z_cols = []


    def reset_reservoir(self, N):
        self.reservoir = WeightedReservoir(N, self.interval, self.policy)

    def update_cost_profile(self, samples):
        self.cost_profile = []
        for i, states in enumerate(self.all_states):
            costs, _ = states.eval(samples, False)
            self.cost_profile.append(costs)

    def detect_redundant_states(self, curr_idx):
        to_delete = []
        if self.epsilon == 0:
            return to_delete
        L = len(self.cost_profile[0])
        for k, state in enumerate(self.all_states):
            if k == curr_idx:
                continue
            distances = []
            ref_profile = self.cost_profile[k]
            for i, profile in enumerate(self.cost_profile):
                if i == k or i in to_delete or i == curr_idx:
                    continue
                d = np.linalg.norm(np.array(profile) - np.array(ref_profile), ord=1) / L
                distances.append(d)
            if len(distances) == 0:
                continue
            d = np.min(distances)
            if d < self.epsilon:
                to_delete.append(k)
        return to_delete

    def delete_states(self, to_delete):
        states = []
        profiles = []
        for i in range(len(self.all_states)):
            if not i in to_delete:
                states.append(self.all_states[i])
                profiles.append(self.cost_profile[i])
        self.all_states = states
        self.cost_profile = profiles

    def add_new_states(self, states, samples):
        #self.update_cost_profile(samples)
        if self.epsilon == 0:
            return list(states.values())

        self.update_cost_profile(samples)
        new_states = []
        L = len(samples)
        for k in states:
            state = states[k]
            new_profile, _ = state.eval(samples, False)
            distances = []
            for profile in self.cost_profile:
                d = np.linalg.norm(np.array(profile) - np.array(new_profile), ord=1) / L
                distances.append(d)
            d = np.min(distances)
            if d > self.epsilon:
                new_states.append(state)
                self.all_states.append(state)
                self.cost_profile.append(new_profile)
        # print("ep:", self.epsilon, ",", len(self.all_states))
        return new_states

    def process_queries(self, new_queries):
        sort_cols = []
        # global sort_cols
        self.reservoir.insert(new_queries)
        new_states = {}
        for p in self.policy:
            if p == 'oracle':
                continue
            samples = self.reservoir.get_samples(p)
            for queries in samples:
                if self.tree_builder.method == 'c':
                    cdf_cols = get_top_columns4(self.tree_builder.cfg, queries)
                    sort_cols = [str(col) for col in cdf_cols]
                    key = ','.join(sort_cols)
                    if key in self.z_cols:
                        continue
                    else:
                        self.z_cols.append(key)
                        # with open(self.col_dir, 'a') as file:
                        # file.write(key + '\n')  # 加入新key
                elif self.tree_builder.method == 'd':
                    cdf_cols = get_top_columns5(self.tree_builder.cfg, queries)
                    sort_cols = [str(col) for col in cdf_cols]
                    key = ','.join(sort_cols)
                    if key in self.z_cols:
                        continue
                    else:
                        self.z_cols.append(key)
                        # with open(self.col_dir, 'a') as file:
                        # file.write(key + '\n')  # 加入新key
                path = "%s/%d.p" % (self.output_dir[p], self.counter[p])
                if self.tree_builder.method == 'c' or self.tree_builder.method == 'd':
                    path += '-label'
                if self.load or os.path.exists(path):
                    #self.counter[p] += 1
                    #continue
                    new_dl = self.tree_builder.load_by_path(
                        path, self.counter[p] == 0, True, sort_cols, self.all_states[0], queries)
                else:
                    print("Computing", path)
                    new_dl = self.tree_builder.compute_optimal_layout(
                        queries, path, self.all_states[0])
                self.counter[p] += 1
                new_states[p] = new_dl

        return self.add_new_states(new_states, self.reservoir.get_profile())
