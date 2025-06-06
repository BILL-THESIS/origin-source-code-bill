import pickle

import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
from itertools import combinations

from scipy.stats import spearmanr


# Union-Find class to handle grouping
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX


def select_cols(data):
    prefix = "java:"
    suffix = "_created"
    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].fillna(0)

    return selected_cols


def filter_cols(data):
    # สร้าง combinations แบบคู่จากคอลัมน์ทั้งหมด
    columns = data.columns
    column_pairs = list(combinations(columns, 2))
    return column_pairs


def calculate_corr(data):
    # คำนวณ Spearman correlation สำหรับแต่ละคู่
    pair_correlations = []
    for col1, col2 in data:
        # คำนวณค่า rank ของแต่ละคอลัมน์
        col1_rank = selected_cols[col1].rank()
        col2_rank = selected_cols[col2].rank()
        # เอาค่า rank มาคำนวณ Spearman correlation
        spearman_rank, p_value = stats.spearmanr(col1_rank, col2_rank)

        pair_correlations.append({
            'col1': col1,
            'col2': col2,
            'spearman_rank': spearman_rank,
            'p_value': p_value
        })

        df_corr = pd.DataFrame(pair_correlations)

    conditions_r = [
        (df_corr['spearman_rank'] <= 0.1),
        (df_corr['spearman_rank'] <= 0.3),
        (df_corr['spearman_rank'] <= 0.5),
        (df_corr['spearman_rank'] <= 0.7),
        (df_corr['spearman_rank'] > 0.7)
    ]

    choices_r = ['no correlation', 'low correlation', 'moderate correlation', 'high correlation',
                 'very high correlation']
    # df_corr['group_r'] = np.select(conditions_r, choices_r)
    df_corr['group_r'] = np.select(conditions_r, choices_r, default='unknown')
    return df_corr


# Function to group coordinates
def group_coordinates_from_df(df):
    uf = UnionFind()

    # Union the pairs from the DataFrame
    for x, y in zip(df['col1'], df['col2']):
        uf.union(x, y)

    # Group the connected components
    groups = {}
    for key in uf.parent:
        root = uf.find(key)
        groups.setdefault(root, set()).add(key)

    # Convert to sorted lists
    return [sorted(group) for group in groups.values()]


if __name__ == "__main__":
    # significant level
    data_significant = pd.read_pickle('../man-whitney/output_man/seatannel_importance.pkl')

    # File paths
    input_filepath = "../output/seatunnal_cut_time.pkl"
    data = pd.read_pickle(input_filepath)

    selected_cols = select_cols(data)
    selected_cols = selected_cols[['java:S2209_created', 'java:S1659_created',
                                   'java:S1181_created', 'java:S100_created',
                                   'java:S101_created', 'java:S3329_created']]
    d = selected_cols.head(5)

    corr_matrix, _ = spearmanr(d)

    metric_column = data_significant['metric']
    selected_columns = selected_cols.columns
    col = set(metric_column).intersection(selected_columns)

    selected_cols = selected_cols[list(col)]
    selected_cols = selected_cols.head(5)

    column_pairs = filter_cols(selected_cols)
    df_corr = calculate_corr(column_pairs)
    df1 = df_corr.iloc[:,:2]
    df_corr = df_corr.round(4)
    # df_corr.to_pickle("output_variable/seatunnal_spearman_rank_all_case.pkl")
    #
    df_corr_high = df_corr[df_corr['group_r'] == 'very high correlation']
    # df_corr_high.to_pickle("output_variable/seatunnel_spearman_rank_high_case.pkl")
