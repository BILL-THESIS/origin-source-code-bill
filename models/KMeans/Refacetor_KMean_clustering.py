import random

import numpy as np
import pandas as pd
import itertools

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from random import shuffle


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

    # step 1


# input: Dataframe X = Original 20 columns
# output: Combination of DataFrame columns
class KMeansCluster:

    def __init__(self, df_original_20_col: pd.DataFrame, chunk_size: int, combi_column_list: list, scarler):
        self.df = df_original_20_col
        self.scarler = scarler
        self.all_combinations = combi_column_list
        self.only_twenty = combi_column_list[:40]
        self.all_sub_combinations_col = [combi_column_list[i:i + chunk_size] for i in
                                         range(0, len(combi_column_list), chunk_size)]
        self.all_sub_combinations_new = list(divide_chunks(combi_column_list, chunk_size))
        self.something(chunk_size)

        # one obj of sub_combinations_col
        self.data_from_combi = [self.df[list(i)] for i in self.all_sub_combinations_col[0]]
        self.data_frames = list(map(lambda subset: self.df[list(subset)], self.all_sub_combinations_col[0]))
        # print(data_frames)

    @staticmethod
    def chunks_r(l, n, is_shuffle=False):
        if is_shuffle:
            random.shuffle(l)
        binning = [None] * n
        for i in range(n):
            binning[i] = []
        for i, s in enumerate(l):
            binning[i % n].append(s)
        return binning

    def something(self, chunk_size):
        self.all_sub_combinations_check = [self.only_twenty[i:i + chunk_size] for i in
                                           range(0, len(self.only_twenty), chunk_size)]

    # functions to divide the combinations of columns in the dataframe df_original_20_col into 8 parts
    # def chunkify(lst, chunk_size):
    #     return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    # def all_sub_combinations(self):
    #     return KMeansCluster.chunkify(self.combinations_list, 8)

    # step 2
    # input: Combination of DataFrame 20 columns to 1 million data sets
    #  output:  MinMax scaler

    def fit_scaler(self):
        for i in self.data_from_combi:
            scaled = self.scarler.fit_transform(i)
            scaled_df = pd.DataFrame(scaled, columns=i.columns)
        return scaled_df

    # step 3
    # input: data frame Minmax scaler
    # output:  Kmeans cluster (score cluster, labels cluster, number cluster)

    def kmeans_cluster(scaled_df):
        kmeans_list = []
        for n in range(2, 5):
            kmeans = KMeans(n_clusters=n, n_init=10)
            kmeans.fit_transform(scaled_df)
            # kmeans.fit(scaled_df)
            score = silhouette_score(scaled_df, kmeans.labels_)
            kmeans_list.append([score, kmeans.labels_, kmeans.n_clusters])
        return kmeans_list

    # def fit_scaler(self, data):
    #     scaled = self.scarler.fit_transform(data)
    #     scaled_df = pd.DataFrame(scaled, columns=data.columns)
    #     return scaled_df


if __name__ == '__main__':

    minmax_scaler = MinMaxScaler()

    # prepare the data frame
    df_original_20_col = pd.read_parquet('seatunnal_20col.parquet')
    df_original_all_col = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
    col_names = df_original_20_col.columns

    # list all combinations of columns in the dataframe X
    all_combinations_list_col = [list(itertools.combinations(col_names, r)) for r in range(2, len(col_names)+1)]

    # amout of columns in the dataframe X
    # all_combianations_col = [itertools.combinations(col_names, r) for r in range(1, len(col_names))]

    # all amout of combinations of columns in the dataframe X 1048554
    all_combinations = list(itertools.chain(*all_combinations_list_col))
    all_combinations = [a for a in all_combinations if len(a) > 0]



    bill = KMeansCluster(df_original_20_col, 8, all_combinations, minmax_scaler)
    c1 = KMeansCluster.chunks_r(all_combinations, 8, False)

    sub_c1 = [c[:10] for c in c1]
    sub_c2 = [c[-10:] for c in c1]

    # loop sub_combinations
    loop_check_sub_combi = [x for sub1 in bill.all_sub_combinations_col for x in sub1]

    s1 = bill.fit_scaler()

    # fit_scaler = bill.fit_scaler()
    # kmeans_results = bill.kmeans_cluster(fit_scaler)
