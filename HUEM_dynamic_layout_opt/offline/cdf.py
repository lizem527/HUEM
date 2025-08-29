from offline.layout import *
from sklearn.cluster import KMeans
#from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import MinMaxScaler
import math
import collections
from tqdm import tqdm


def decimalToBinary(n, max_bits):
    raw = bin(n).replace("0b", "")
    return '0' * (max_bits - len(raw)) + raw


def get_top_columns4(cfg, workload):
    cnt = {}
    q1 = []
    q2 = []
    q3 = []
    for q in workload:
        leaves = q.get_leaves()
        for leaf in leaves:
            list = [val for val in leaf.vals if not np.isnan(val)]
            t = list[0]
            if leaf.col == 'lat':
                q1.append(t)
            elif leaf.col == 'lon':
                q2.append(t)
            elif leaf.col == 'spei':
                q3.append(t)
    # Heuristic: date and numeric columns before categorical columns
    q = [sum(arr) / len(arr) if arr else 0 for arr in (q1, q2, q3)]
    return q


def code_to_label(labels, k, now_centers, X, total_center):
    cluster_centers = now_centers

    # 计算每行数据的CFD
    distances = []
    print("computing CDF")
    for i, label in enumerate(labels):
        center = cluster_centers[label]
        distance_to_center = np.linalg.norm(X[i] - center)
        distance_center_to_total = np.linalg.norm(center - total_center)
        total_distance = distance_to_center + distance_center_to_total
        distances.append(total_distance)

    distances = np.array(distances)

    indices = np.argsort(distances)
    part_size = len(distances) // k + 1
    bids = np.zeros(len(distances))
    for i in range(k):
        if i == k - 1:
            bids[indices[i * part_size:]] = i
        else:
            bids[indices[i * part_size:(i + 1) * part_size]] = i

    return bids


class CDF(Layout):
    def __init__(self, df, cfg, k, max_bits=24, workload=None, verbose=True, init=None):
        super().__init__(df, cfg, k)
        self.cluster_centers = None
        self.codes = None
        self.workload = workload
        self.init = init
        self.vals = None
        self.total_center = None
        self.max_bits = max_bits
        self.bits = self.max_bits
        self.verbose = verbose
        self.sort_cols = ['lat', 'lon', 'spei']

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

    def compute_center(self, q, df):
        df_row = pd.DataFrame([q], columns=df.columns)
        df = pd.concat([df, df_row], ignore_index=True)
        center = []
        for i, col in enumerate(df.columns):
            vals = df[col].values
            # Normalize numeric values to integers in [0, 2^(max_bits)]
            if self.types[i] == "num":
                scaler = MinMaxScaler()
                vals = vals.reshape(-1, 1)
                scaler.fit(vals)
                new_vals = np.squeeze(scaler.transform(vals)) * (math.pow(2, self.max_bits) - 1)
                new_vals = new_vals.astype(int)
                last_val = new_vals[-1]
            center.append(last_val)
        return center

    def _get_codes(self):
        X = self.vals
        print("X")
        k_length = 2 #self.k * 5
        # 使用K-Means进行聚类
        kmeans = KMeans(n_clusters=k_length, random_state=42)

        with tqdm(total=1, desc="聚类进度") as pbar:
            labels = kmeans.fit_predict(X)
            pbar.update(1)  # 更新进度条

        # 统计每个类的数据数量
        cluster_sizes = np.bincount(labels)
        print("聚类数", cluster_sizes)

        self.cluster_centers = kmeans.cluster_centers_

        return labels

    def prepare_code(self):
        df = self._remove_unused_cols(self.cfg, self.df)
        self.vals = self._transform(df)
        if self.init is not None:
            self.codes = self.init.codes
            self.cluster_centers = self.init.cluster_centers
        else:
            self.codes = self._get_codes()

        if self.workload is not None:
            q = get_top_columns4(self.cfg, self.workload)
            self.total_center = self.compute_center(q, df)
            # print("former,",self.total_center)
            for i, ave_q in enumerate(q):
                if q[i] == 0:
                    original_center = np.mean(self.init.cluster_centers, axis=0)
                    self.total_center[i] = original_center[i]
            print("compute_total_center,", self.total_center)
        else:
            self.total_center = np.mean(self.cluster_centers, axis=0)
            print("orignal_total_center,", self.total_center)

    def _get_labels(self):
        self.prepare_code()
        self.labels = code_to_label(self.codes, self.k, self.cluster_centers, self.vals, self.total_center)

    def make_partitions(self):
        self._get_labels()
        self.load_by_labels(self.labels)

    def save_by_path(self, path):
        self.path = path
        pickle.dump(self.labels, open(self.path, "wb"))

    def load_by_path(self, path):
        labels = pickle.load(open(path, "rb"))
        self.path = path
        self.load_by_labels(labels)
