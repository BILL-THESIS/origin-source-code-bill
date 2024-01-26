#!/usr/local/bin/python3
import time
from datetime import timedelta
import pandas as pd
from nltk import word_tokenize, PorterStemmer
from sklearn.metrics._dist_metrics import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import requests

import os


# input file combianation part
def read_parquet():
    directory_path = '../../models/KMeans/combia2'
    parquet_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    global df_list
    df_list = []
    for parquet_file in parquet_files:
        file_path = os.path.join(directory_path, parquet_file)
        df = pd.read_parquet(file_path)
        df_list.append(df)
    return df_list


df_original = read_parquet()


def scale_data(df):
    scaler_list = []
    for data_scaler in df:
        scaler = MinMaxScaler()
        x_scaler = scaler.fit_transform(data_scaler)
        scaled = pd.DataFrame(x_scaler, columns=data_scaler.columns)
        scaler_list.append(scaled)
    return scaler_list


# scaler_list = scale_data(df_original)


def kmeans_cluster(df):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")
    score2 = 0
    score3 = 0
    score4 = 0
    score2_list = []
    score3_list = []
    score4_list = []

    for data_scaler in scale_data(df):
        for i in df_list:
            for n_clusters in range(2, 5):
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaler)
                clusters = silhouette_score(data_scaler, kmeans.labels_)

                if n_clusters == 2:
                    value_2 = {
                        'df': i,
                        'cluster_labels': cluster_labels,
                        'cluster': kmeans,
                        'score': clusters
                    }
                    score2_list.append(value_2)
                    score2 = pd.DataFrame(score2_list)
                    score2.drop_duplicates(subset=['df'], inplace=True)
                    print("\n")
                if n_clusters == 3:
                    value_3 = {
                        'df': i,
                        'cluster_labels': cluster_labels,
                        'cluster': kmeans,
                        'score': clusters,
                    }
                    score3_list.append(value_3)
                    score3 = pd.DataFrame(score3_list)
                    score3.drop_duplicates(subset=['df'], inplace=True)
                    # print("score3_list", score3)
                if n_clusters == 4:
                    value_4 = {
                        'df': i,
                        'cluster_labels': cluster_labels,
                        'cluster': kmeans,
                        'score': clusters
                    }
                    score4_list.append(value_4)
                    score4 = pd.DataFrame(score4_list)
                    score4.drop_duplicates(subset=['df'], inplace=True)
                    # print("score4_list", score4)

    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    return score2, score3, score4


# Limit to just 120000 rows
list_df_list = df_list[:4]

# multiprocessing part
cpus = 2
# cpus = multiprocessing.cpu_count()
# paresed data is data files combianation part from list df
# parsed_description_split = np.array_split(list_df_list, cpus)
parsed_description_split = [[list_df_list[0], list_df_list[2]]
                            , [list_df_list[1], list_df_list[3]]
                            , [list_df_list[1], list_df_list[3]]
                            # , [list_df_list[5], list_df_list[6]]
                            # , [list_df_list[7], list_df_list[8]]
                            ]
print(f"Processing files in directory {list_df_list} using {cpus} CPU cores")
# print("Number of cpus: ", cpus)
print("Number of splits: ", len(parsed_description_split))


def get_df_cluster(df):
    start = time.time()
    get_cluster = kmeans_cluster(df)
    end = time.time()
    get_time = end - start
    print("Total time: ", get_time)
    return get_cluster


# with multiprocessing.Pool(cpus) as pool:
with multiprocessing.pool.ThreadPool(cpus) as pool:
    parsed_description_split = pool.map(get_df_cluster, parsed_description_split)
    print(parsed_description_split)

    # parsed_description = pd.concat(parsed_description_split)
    # print(parsed_description.head())
