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

df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
path2 = os.path.abspath("../../output/cluster2")
path3 = os.path.abspath("../../output/cluster3")
path4 = os.path.abspath("../../output/cluster4")


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


df_combaination = read_parquet()


def scale_data(df):
    scaler_list = []
    for data_scaler in df:
        scaler = MinMaxScaler()
        x_scaler = scaler.fit_transform(data_scaler)
        scaled = pd.DataFrame(x_scaler, columns=data_scaler.columns)
        scaler_list.append(scaled)
    return scaler_list


def kmeans_cluster(df):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    list_label = []
    for data_scaler in scale_data(df):
        # print("data_scaler", data_scaler)
        for i in df_list:
            for n_clusters in range(2, 5):
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaler)
                df_cluster_labels = pd.DataFrame(cluster_labels)
                clusters = silhouette_score(data_scaler, kmeans.labels_).__round__(4)
                list_label.append([i, df_cluster_labels, clusters, n_clusters])
                # print("list_label", list_label)
                df_lables = pd.DataFrame(list_label)

                for j,row in df_lables.iterrows():

                    df_concat_col = pd.DataFrame(row[0])
                    # print("Dataframe 1", df_concat_col)
                    df_concat_col['cluster_labels'] = row[1]
                    df_concat_col['score'] = row[2]
                    df_concat_col['clusters'] = row[3]

                    if df_concat_col['clusters'].values[0] == 2:
                        merged_df_original2 = pd.concat([df_original['total_time'], df_concat_col], axis=1).reindex(df_concat_col.index)
                        print("merged_df_original2", merged_df_original2)
                        merged_df_original2.to_parquet(path2 + f'/{row[0].columns.to_list}_{row[3]}.parquet')
                    if df_concat_col['clusters'].values[0] == 3:
                        merged_df_original3 = pd.concat([df_original['total_time'], df_concat_col], axis=1).reindex(df_concat_col.index)
                        merged_df_original3.to_parquet(path3 + f'/{row[0].columns.to_list}_{row[3]}.parquet')
                    if df_concat_col['clusters'].values[0] == 4:
                        merged_df_original4 = pd.concat([df_original['total_time'], df_concat_col], axis=1).reindex(df_concat_col.index)
                        merged_df_original4.to_parquet(path4 + f'/{row[0].columns.to_list}_{row[3]}.parquet')
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time : {result_time}")
    return merged_df_original2, merged_df_original3, merged_df_original4


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
    print("Total time get cluster: ", get_time)
    return get_cluster


# with multiprocessing.Pool(cpus) as pool:
with multiprocessing.pool.ThreadPool(cpus) as pool:
    parsed_description_split = pool.map(get_df_cluster, parsed_description_split)
    print(parsed_description_split)
