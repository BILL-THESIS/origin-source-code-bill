import multiprocessing
import random
import time
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import platform

import numpy as np
import pandas as pd
import itertools
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import threadpoolctl
from datetime import datetime
import pickle

from joblib import Parallel, delayed


# input: Dataframe X = Original 20 columns
# output: Combination of DataFrame columns
class KMeansCluster:

    def __init__(self, df_original_20_col: pd.DataFrame, scarler, sub_combination , thread_limit):
        self.df = df_original_20_col
        self.scarler = scarler
        self.all_combinations_list_col = [list(itertools.combinations(self.df.columns, r)) for r in
                                          range(2, len(self.df.columns) + 1)]
        self.a = list(itertools.chain(*self.all_combinations_list_col))
        self.all_combinations = [a for a in self.a if len(a) > 0]

        self.sub_combinations = sub_combination
        self.thread_limits = thread_limit

        # self.scarler_fit_transform = self.scarler.fit_transform(self.df)
        # self.scaled_df = pd.DataFrame(self.scarler_fit_transform, columns=self.df.columns)

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
        del scaled
        return scaled_df

    # step 3
    # input: data frame Minmax scaler
    # output:  Kmeans cluster (score cluster, labels cluster, number cluster)

    def kmeans_cluster(self, scaled_df_fit_transform):

        """
            Performs K-means clustering with silhouette score evaluation,
            using joblib's Loky backend for potential parallelism with OpenMP.
        """
        results = []

        # Set start method for multiprocessing
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn')

        try:
            # Control OpenMP threads using threadpoolctl (if available)
            threadpoolctl.threadpool_limits(limits=self.thread_limits)
            # Adjust thread limit as needed
        except ImportError:
            pass  # If threadpoolctl is not available, rely on environment variables

        for n in range(2, 5):
            kmeans = KMeans(n_clusters=n, n_init=10)
            with joblib.parallel_backend('loky'):  # Use Loky backend for potential OpenMP
                kmeans.fit(scaled_df_fit_transform)
                score = silhouette_score(scaled_df_fit_transform, kmeans.labels_)
                results.append((n, score, kmeans.labels_))
        return results

    def loop_cluster(self, sub_list):
        results = []
        count = 0
        for i in sub_list:
            # print("I ::", i)
            for j in i:
                # print("J ::", j)
                # fit = self.fit_scaler(self.df[list(j)])
                fit = self.fit_scaler(self.df[list(j)])
                # print("FIT ::", fit)
                k = self.kmeans_cluster(fit)
                k_list = ({
                    'df': j,
                    '2': k[0],
                    '3': k[1],
                    '4': k[2]})
                results.append(k_list)
                # print("K ::", k_list)
                print("Results ::", len(results))

        return results



if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    minmax_scaler = MinMaxScaler()
    cpus = 3

    # prepare the data frame
    df_original_20_col = pd.read_parquet('../../output/seatunnal_20col.parquet')
    df_original_20_col = df_original_20_col.drop(
        columns=['D_change', 'B_change', 'CP_change', 'C_change', 'OOA_change'])
    df_original_all_col = pd.read_parquet('../../../Sonar/seatunnel_all_information.parquet')
    col_names = df_original_20_col.columns

    all_combinations_list_col = [list(itertools.combinations(df_original_20_col.columns, r)) for r in
                                 range(2, len(df_original_20_col.columns) + 1)]

    a1 = list(itertools.chain(*all_combinations_list_col))
    all_combinations = [a for a in a1 if len(a) > 0]

    check = KMeansCluster.chunks(all_combinations, 8, False)
    sub_c1 = [c[:10] for c in check]
    sub_c2 = [c[-10:] for c in check]

    bill = KMeansCluster(df_original_20_col, minmax_scaler, check, cpus)
    k = bill.loop_cluster(check)

    if platform.system() == 'Linux':
        # Code to execute if running on Linux
        print("Running on Linux")
        with multiprocessing.pool.Pool(processes=cpus, maxtasksperchild=1) as pool:
            parsed_split = pool.map(bill.loop_cluster, sub_c2, 3)
            print("Thread Pool of parsed_data ::", parsed_split)

    elif platform.system() == 'Darwin':
        # Code to execute if running on macOS (Darwin is the underlying OS for macOS)
        print("Running on macOS")
        with multiprocessing.pool.Pool(cpus) as pool:
            # r = pool.apply_async(bill.loop_cluster, (sub_c2, 3))
            # print("Thread Pool of parsed_data ::", pool.starmap_async(bill.loop_cluster, sub_c2, 3))
            parsed_description_split = pool.starmap_async(bill.loop_cluster, check)
            print("Thread Pool of parsed_data ::", parsed_description_split)

    else:
        # Code to execute if running on another OS
        print("Running on an unknown or unsupported OS")

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
