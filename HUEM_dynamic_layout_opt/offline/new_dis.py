from offline.layout import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import math
import collections
from tqdm import tqdm
import time


def decimalToBinary(n, max_bits):
    raw = bin(n).replace("0b", "")
    return '0' * (max_bits - len(raw)) + raw


def get_q(query,sort_cols):
    col_values = {col: [] for col in sort_cols}
    col_values_min = {col: [] for col in sort_cols}
    # print("i",i,"type",type(query))
    leaves = query.get_leaves()
    for leaf in leaves:
        list = [val for val in leaf.vals if not np.isnan(val)]
        t = list[0]
        if leaf.col in col_values:
            if np.isnan(leaf.vals[1]):
                col_values_min[leaf.col].append(t)
            col_values[leaf.col].append(t)
    q = [sum(arr) / len(arr) if arr else 0 for arr in (col_values[col] for col in sort_cols)]

    q_min = [sum(arr) / len(arr) if arr else 0 for arr in (col_values_min[col] for col in sort_cols)]
    return q, q_min


def get_top_columns5(cfg, workload):
    cnt = {}
    sort_cols = cfg["num_cols"]
    col_values = {col: [] for col in sort_cols}
    for q in workload:
        leaves = q.get_leaves()
        for leaf in leaves:
            list = [val for val in leaf.vals if not np.isnan(val)]
            t = list[0]
            if leaf.col in col_values:
                col_values[leaf.col].append(t)
    # Heuristic: date and numeric columns before categorical columns
    q = [sum(arr) / len(arr) if arr else 0 for arr in (col_values[col] for col in sort_cols)]
    return q


def code_to_label(distances, k):
    indices = np.argsort(distances)
    part_size = len(distances) // k + 1
    bids = np.zeros(len(distances))
    for i in range(k):

        if i == k - 1:
            bids[indices[i * part_size:]] = i
        else:
            bids[indices[i * part_size:(i + 1) * part_size]] = i

    return bids


class DIS(Layout):
    def __init__(self, df, cfg, k, max_bits=24, workload=None, verbose=True, init=None):
        super().__init__(df, cfg, k)
        self.meta = None
        self.codes = None
        self.init = init
        self.workload = workload
        self.max_bits = max_bits
        self.bits = self.max_bits
        self.verbose = verbose
        self.sort_cols = cfg["num_cols"]
        self.scaler = []
        self.n = 1 #block numbers for tabel l
        self.df2 = self._remove_unused_cols(self.cfg, self.df)
        self.vals = self._transform(self.df2)
        if self.workload is not None:
            q = get_top_columns5(self.cfg, self.workload)
            self.center, _ = np.array(self.compute_center(q, q, self.df2))

        else:
            self.center = self._get_kcenter()

        self.codes = np.array(self._get_codes())

    def _remove_unused_cols(self, cfg, df):
        # Extract columns that are involved in the query
        cols = []
        idx = []
        for i, col in enumerate(df.columns):
            if not col in self.sort_cols:
                continue
            cols.append(col)
            idx.append(i)
        df = df[cols]
        # Store column types
        types = []
        for col in df.columns:
            if col in cfg["num_cols"]:
                types.append("num")
            elif col in cfg["cat_cols"]:
                types.append("cat")
            else:
                types.append("date")
        df = df.reindex(columns=self.sort_cols)
        self.d = len(cols)
        self.types = types
        return df

    def _transform(self, df):
        """Prepare columns values for kmeans"""
        new_df = collections.OrderedDict()
        for i, col in enumerate(df.columns):
            vals = df[col].values
            # Normalize numeric values to integers in [0, 2^(max_bits)]
            if self.types[i] == "num":

                scaler = MinMaxScaler()
                vals = vals.reshape(-1, 1)
                scaler.fit(vals)
                self.scaler.append(scaler)
                new_vals = np.squeeze(scaler.transform(vals)) * (math.pow(2, self.max_bits) - 1)
                new_vals = new_vals.astype(int)
            # Transform strings into binary by keeping track of unique values
            else:
                # Integer encode strings via alphabetical order
                # This is needed since date columns have orders
                mapping = {}
                unique = sorted(list(set(vals)))
                for val in unique:
                    mapping[val] = len(mapping)
                new_vals = []
                for val in vals:
                    new_vals.append(mapping[val])
                num_bits = int(math.log2(len(mapping)))
                if math.pow(2, num_bits) < len(mapping):
                    num_bits += 1
                self.bits = max(self.max_bits, num_bits)
            new_df[col] = new_vals
        new_dataframe = pd.DataFrame.from_dict(new_df)
        return new_dataframe.values

    def compute_center(self, q1, q2, df):
        df_row1 = pd.DataFrame([q1], columns=df.columns)
        df_row2 = pd.DataFrame([q2], columns=df.columns)
        df = pd.concat([df_row1, df_row2], ignore_index=True)
        center = []
        cen2 = []
        for i, col in enumerate(df.columns):
            vals = df[col].values
            # Normalize numeric values to integers in [0, 2^(max_bits)]
            if self.types[i] == "num":
                vals = vals.reshape(-1, 1)
                scaler = self.scaler[i]
                new_vals = np.squeeze(scaler.transform(vals)) * (math.pow(2, self.max_bits) - 1)
                new_vals = new_vals.astype(int)
                val1 = new_vals[0]
                val2 = new_vals[-1]
                center.append(val1)
                cen2.append(val2)
        return center, cen2


    def _get_kcenter(self):
        X = self.vals
        print("X")
        k_length = 2  # self.k * 5
        # use K-Means cluster
        kmeans = KMeans(n_clusters=k_length, random_state=42)

        with tqdm(total=1, desc="cluster progress") as pbar:
            labels = kmeans.fit_predict(X)
            pbar.update(1)  # update progress

        center = np.mean(kmeans.cluster_centers_, axis=0)

        return center

    def _get_codes(self):
        # update dis for each data
        distances = []
        print("computing distance")
        for i in range(self.N):
            distance_to_center = np.linalg.norm(self.vals[i] - self.center)
            distances.append(distance_to_center)

        distances = np.array(distances)

        return distances

    def _get_labels(self):
        self.labels = code_to_label(self.codes, self.k)

    def compute_meta(self):
        metas = []
        for i in range(self.k):
            pid = int(i)
            filtered_codes = self.codes[self.labels == pid]
            max_part = np.max(filtered_codes)
            min_part = np.min(filtered_codes)
            meta = [max_part, min_part, filtered_codes.size]
            metas.append(meta)
        return metas

    def make_partitions(self):
        self._get_labels()
        self.load_by_labels(self.labels)

    def save_by_path(self, path):
        self.path = path
        pickle.dump(self.labels, open(self.path, "wb"))
        if self.meta is None:
            self.meta = self.compute_meta()
            #print("len meta:", len(self.meta))

    def load_by_path(self, path):
        labels = pickle.load(open(path, "rb"))
        self.path = path
        self.load_by_labels(labels)

        if self.meta is None:
            self.meta = self.compute_meta()


    def eval(self, queries, avg=True):
        # compute cost
        read = [0] * len(queries)
        read_pids = []
        # time1 = time.time()

        for i, query in enumerate(queries):

            q, q_min = get_q(query, self.sort_cols)

            cen1, cen2 = np.array(self.compute_center(q, q_min, self.df2))
            # compute Euclidean distance
            R = np.linalg.norm(cen1 - cen2)
            dis = np.linalg.norm(self.center - cen1)
            # print("print distances:", R, dis)
            d_max = R + dis
            if R < dis:
                d_min = dis - R
            else:
                d_min = 0
            pids = []

            if self.init is not None:
                # part tabel l can search
                if d_max <= self.meta[self.n-1][0]:
                    for t in range(self.n):
                        pid = int(t)
                        # print("pid,",pid,self.meta[pid])
                        if self.init.meta[pid][0] >= d_min and self.init.meta[pid][1] <= d_max:
                            read[i] += self.meta[pid][2]
                            pids.append(pid)
                    read_pids.append(pids)
                # else full
                else:
                    dis2 = np.linalg.norm(self.init.center - cen1)
                    d_max = R + dis2
                    if R < dis2:
                        d_min = dis2 - R
                    else:
                        d_min = 0
                    for t in range(self.k):
                        pid = int(t)
                        if self.init.meta[pid][0] >= d_min and self.init.meta[pid][1] <= d_max:
                            read[i] += self.meta[pid][2]
                            pids.append(pid)
                    read_pids.append(pids)

        read = np.array(read) * 1.0 / self.N
        if avg:
            return np.average(read), read_pids
        else:
            return read, read_pids
