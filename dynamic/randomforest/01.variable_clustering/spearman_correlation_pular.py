import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations


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
    df_corr['group_r'] = np.select(conditions_r, choices_r)
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
    data_significant = pd.read_pickle('../../output/pulsar_quatile_significant.pkl')

    # File paths
    input_filepath = "../../output/pulsar_compare.pkl"
    data = pd.read_pickle(input_filepath)


    selected_cols = select_cols(data)

    # คำนวณ Spearman correlation
    data_corr = data.corr(method='spearman')

    column_pairscolumn_pairs = filter_cols(selected_cols)
    df_corr = calculate_corr(column_pairscolumn_pairs)

    df_corr_high = df_corr[df_corr['group_r'] == 'very high correlation']

    # Apply the function
    result_group = group_coordinates_from_df(df_corr_high[['col1', 'col2']])


    # data frame for each group
    data_group1 = data[result_group[0]]
    data_group2 = data[result_group[1]]
    data_group3 = data[result_group[2]]
    data_group4 = data[result_group[3]]
    data_group5 = data[result_group[4]]
    data_group6 = data[result_group[5]]
    data_group7 = data[result_group[6]]

    # สร้าง DataFrame จากผลลัพธ์
    result1_significant = data[result_group[0]].loc[:, data[result_group[0]].columns.isin(data_significant['metric'])]
    result2_significant = data[result_group[1]].loc[:, data[result_group[1]].columns.isin(data_significant['metric'])]
    result3_significant = data[result_group[2]].loc[:, data[result_group[2]].columns.isin(data_significant['metric'])]
    result4_significant = data[result_group[3]].loc[:, data[result_group[3]].columns.isin(data_significant['metric'])]
    result5_significant = data[result_group[4]].loc[:, data[result_group[4]].columns.isin(data_significant['metric'])]
    result6_significant = data[result_group[5]].loc[:, data[result_group[5]].columns.isin(data_significant['metric'])]
    result7_significant = data[result_group[6]].loc[:, data[result_group[6]].columns.isin(data_significant['metric'])]

    # เลือกมาแค่ 1 ตัวสำหรับแต่ละ group
    result_final = pd.concat([data[group[0]] for group in result_group], axis=1).fillna(0)

    result_final['total_time'] = data['total_time']
    result_final.to_pickle('../../output/pulsar_correlation.pkl')
    result1_significant.to_pickle('../../output/pulsar_correlation_group1_significant.pkl')
