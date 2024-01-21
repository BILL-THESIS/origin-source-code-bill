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


def read_parquet():
    directory_path = '../../models/KMeans/combia2-copy'
    parquet_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
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

df_scaler = scale_data(df_original)

def kmeans_cluster(df):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    score2_list = []
    score3_list = []
    score4_list = []

    for data_scaler in scale_data(df):
        for n_clusters in range(2, 5):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(data_scaler)
            clusters = silhouette_score(data_scaler, kmeans.labels_)

            if n_clusters == 2:
                value_2 = {
                    'df': data_scaler,
                    'cluster_labels': cluster_labels,
                    'cluster': kmeans,
                    'score': clusters
                }
                score2_list.append(value_2)
                score2 = pd.DataFrame(score2_list)
                print("\n")
            if n_clusters == 3:
                value_3 = {
                    'df': data_scaler,
                    'cluster_labels': cluster_labels,
                    'cluster': kmeans,
                    'score': clusters
                }
                score3_list.append(value_3)
                score3 = pd.DataFrame(score3_list)
                # print("score3_list", score3)
            if n_clusters == 4:
                value_4 = {
                    'df': data_scaler,
                    'cluster_labels': cluster_labels,
                    'cluster': kmeans,
                    'score': clusters
                }
                score4_list.append(value_4)
                score4 = pd.DataFrame(score4_list)
                # print("score4_list", score4)

    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    return score2, score3, score4


df = kmeans_cluster(df_original)
df = pd.DataFrame(df)
print(df)
