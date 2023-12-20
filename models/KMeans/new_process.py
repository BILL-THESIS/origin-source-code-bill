#!/usr/local/bin/python3
import time
from datetime import timedelta
import pandas as pd
from sklearn.metrics._dist_metrics import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os


# files_path = directory_path
# directory_path_scaled = '../../models/scaleds'

def read_parquet():
    directory_path = '../../models/KMeans/combia2'
    parquet_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    df_list = []
    for parquet_file in parquet_files:
        file_path = os.path.join(directory_path, parquet_file)
        df = pd.read_parquet(file_path)
        df_list.append(df)
    return df_list


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
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time))
    print(f"Start to normalize cluster at: {start_time_gmt}")

    result_dict = {}

    url = 'https://notify-api.line.me/api/notify'
    token_line = "dsIcr3W7g1oMFH5XurbULg2AWfE9xsLAAjchWfFrxnm"
    headers = {'content-type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer ' + token_line}

    for n_clusters in range(2, 5):
        cluster_results = []

        for data_scaler in scale_data(df):
            kmeans = KMeans(n_clusters=n_clusters)
            print(kmeans)
            print("/n")
            cluster_labels = kmeans.fit_predict(data_scaler)
            print(cluster_labels)
            print("/n")
            clusters = silhouette_score(data_scaler, cluster_labels)
            print(clusters)
            line = requests.post(url, headers=headers, data={'data combia': clusters})
            print(line.text)
            value = {
                'col_name': data_scaler.columns,
                'cluster_labels': cluster_labels,
                'cluster': kmeans,
                'score': clusters
            }

            cluster_results.append(value)

        result_dict[f'clusters_{n_clusters}'] = pd.DataFrame(cluster_results)

    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.strftime("%H:%M:%S", time.gmtime(result_time))
    print(f"Total time: {result_time_gmt}")
    return result_dict
    # return result_dict['score2'], result_dict['score3'], result_dict['score4']

# def kmeans_cluster(df):
#     start_time = time.time()
#     start_time_gmt = time.gmtime(start_time)
#     start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
#     print(f"start to normalize cluster at: {start_time_gmt}")
#
#     score2_list = []
#     score3_list = []
#     score4_list = []
#
#     for data_scaler in scale_data(df):
#         for n_clusters in range(2, 5):
#             kmeans = KMeans(n_clusters=n_clusters)
#             cluster_labels = kmeans.fit_predict(data_scaler)
#             clusters = silhouette_score(data_scaler, kmeans.labels_)
#
#             if n_clusters == 2:
#                 value_2 = {
#                     'col_name': data_scaler.columns,
#                     'cluster_labels': cluster_labels,
#                     'cluster': kmeans,
#                     'score': clusters
#                 }
#                 score2_list.append(value_2)
#                 score2 = pd.DataFrame(score2_list)
#                 print("\n")
#             if n_clusters == 3:
#                 value_3 = {
#                     'col_name': data_scaler.columns,
#                     'cluster_labels': cluster_labels,
#                     'cluster': kmeans,
#                     'score': clusters
#                 }
#                 score3_list.append(value_3)
#                 score3 = pd.DataFrame(score3_list)
#                 # print("score3_list", score3)
#             if n_clusters == 4:
#                 value_4 = {
#                     'col_name': data_scaler.columns,
#                     'cluster_labels': cluster_labels,
#                     'cluster': kmeans,
#                     'score': clusters
#                 }
#                 score4_list.append(value_4)
#                 score4 = pd.DataFrame(score4_list)
#                 # print("score4_list", score4)
#
#     end_time = time.time()
#     result_time = end_time - start_time
#     result_time_gmt = time.gmtime(result_time)
#     result_time = time.strftime("%H:%M:%S", result_time_gmt)
#     print(f"Total time: {result_time}")
#     return score2, score3, score4
