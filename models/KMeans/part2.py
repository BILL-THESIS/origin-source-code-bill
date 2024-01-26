#!/usr/local/bin/python3
import time
from datetime import timedelta, datetime
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import timeit
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = '../../models/KMeans/combia2-copy'
path2 = Path(os.path.abspath("../../models/KMeans/cluster2"))
df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

parquet_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]

scaler = MinMaxScaler()
scaled_dataframes = []
scores = []
labels = []
result_dfs = []

list_time_loop = []

url = 'https://notify-api.line.me/api/notify'
token = "H5TmIeN7Sj7FviOgCJFK4HKE9jBw5h6kdoY6nmdSdpL"
headers = {'content-type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer ' + token}

start_time = time.time()
start_time_gmt = time.gmtime(start_time)
start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
print("Start time ::", start_time_gmt)

for csv_file in parquet_files:
    file_path = os.path.join(directory_path, csv_file)
    # print("file :::" , file_path)
    variable_name = os.path.splitext(csv_file)[0]
    # print("Var ::" , variable_name)
    df_col_combined = pd.read_parquet(file_path)
    # print("DF ::" , df_col_combined)

    scaled_data = scaler.fit_transform(df_col_combined)
    # print("Scaled :::", scaled_data)

    scaled_df = pd.DataFrame(scaled_data, columns=df_col_combined.columns)

    for n_clusters in range(2, 3):  # 11
        km = KMeans(n_clusters=n_clusters)
        # print("KM :::", km)
        km.fit(scaled_df)
        sil_avg = silhouette_score(scaled_df, km.labels_).round(4)
        # print("SCORES :::", scores)

        cluster_labels = km.fit_predict(scaled_df)
        df_cluster_labels = pd.DataFrame(cluster_labels)
        print("CLUSTER :::", cluster_labels)

        labels.append([df_col_combined, df_cluster_labels, sil_avg, n_clusters])
        # print(labels)
        df_lables = pd.DataFrame(labels)

        for i, row in df_lables.iterrows():

            print("ROW 3 ", row[2])
            print('\n')

            print("ROW 4 ", row[3])
            print('\n')

            df1 = pd.concat([row[0], row[1]], axis=1)
            # print("==========================")

            df1 = df1.rename(columns={0: f'{row[0].columns.to_list()}_{row[3]}'})
            df1['scored'] = row[2]
            df1['clusters'] = row[3]

            # start_time_loop = time.time()
            # start_time_gmt_loop = time.gmtime(start_time_loop)
            # start_time_gmt_loop = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt_loop)
            # print("Start time loop::", start_time_gmt_loop)

            if df1['clusters'].values[0] == 2:
                merged_df3 = pd.concat([df_original['total_time'], df1], axis=1).reindex(
                    df1.index)

end_time = time.time()
result_time = end_time - start_time
result_time_gmt = time.gmtime(result_time)
result_time_gmt = time.strftime("%H:%M:%S", result_time_gmt)
print(f"Total time: {result_time}")
print("Time gmt :::", result_time_gmt)
