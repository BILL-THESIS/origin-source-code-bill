import multiprocessing
import random
import time
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd
import itertools

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# input: Dataframe X = Original 20 columns
# output: Combination of DataFrame columns
class KMeansCluster:

    def __init__(self, df_original_20_col: pd.DataFrame, scarler ):
        self.df = df_original_20_col
        self.scarler = scarler
        self.all_combinations_list_col = [list(itertools.combinations(self.df.columns, r)) for r in
                                          range(2, len(self.df.columns) + 1)]
        self.a = list(itertools.chain(*self.all_combinations_list_col))
        self.all_combinations = [a for a in self.a if len(a) > 0]

    @staticmethod
    def chunks(list_combi, n, is_shuffle=False):
        if is_shuffle:
            random.shuffle(list_combi)
        binning = [None] * n
        for i in range(n):
            binning[i] = []
        for i, s in enumerate(list_combi):
            binning[i % n].append(s)
        return binning

    # step 2
    # input: Combination of DataFrame 20 columns to 1 million data sets
    #  output:  MinMax scaler

    def fit_scaler(self, df):
        scaled = self.scarler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)
        return scaled_df

    # step 3
    # input: data frame Minmax scaler
    # output:  Kmeans cluster (score cluster, labels cluster, number cluster)

    def kmeans_cluster(self, scaled_df_fit_transform):
        results = []
        for n in range(2, 5):
            kmeans = KMeans(n_clusters=n, n_init=10)
            kmeans.fit_transform(scaled_df_fit_transform)
            score = silhouette_score(scaled_df_fit_transform, kmeans.labels_)
            results.append((n, score, kmeans.labels_))
        return results



if __name__ == '__main__':
    start = time.time()
    print("start time ::", start)

    minmax_scaler = MinMaxScaler()
    cpus = 4

    # prepare the data frame
    df_original_20_col = pd.read_parquet('seatunnal_20col.parquet')
    df_original_all_col = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
    col_names = df_original_20_col.columns

    bill = KMeansCluster(df_original_20_col, minmax_scaler)

    c1 = bill.chunks(bill.all_combinations, 8, False)

    sub_c1 = [c[:10] for c in c1]
    sub_c2 = [c[-10:] for c in c1]


    def k2(sub_list):
        results = []
        for i in sub_list:
            for j in i:
                s = bill.fit_scaler(df_original_20_col[list(j)])
                k = bill.kmeans_cluster(s)
                k_list = ({
                    'df': j,
                    '2': k[0],
                    '3': k[1],
                    '4': k[2]})
                results.append(k_list)
        return results

    do = k2(sub_c2)
    do_chunks = bill.chunks(do, 8, False)
    # do_list = list(do.items() for do in do)


    parsed_description_split = [
        [list(sub_c2[0]), list(sub_c2[1])],
        [list(sub_c2[2]), list(sub_c2[3])],
        [list(sub_c2[4]), list(sub_c2[5])],
        [list(sub_c2[6]), list(sub_c2[7])]
    ]

    # with multiprocessing.pool.ThreadPool(cpus) as pool:
        # obj is not callable
        # type object 'KMeansCluster' has no attribute 'k2'
        # parsed_data = pool.starmap(k2, sub_c2)
        # parsed_data = pool.starmap(do, parsed_description_split)
        # print(f"Thread Pool of parsed_data ::", parsed_data)

    with ThreadPoolExecutor(max_workers=cpus) as executor:
        parsed_description_split = list(executor.map(k2, parsed_description_split))
        print("Thread Pool of parsed_data ::", parsed_description_split)

    end = time.time()
    total_time = end - start
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
