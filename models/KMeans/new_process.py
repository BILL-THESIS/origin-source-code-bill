#!/usr/local/bin/python3
import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
import itertools
import numpy as np
from part1 import X

df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
path2 = os.path.join(os.path.abspath('../../output'), 'cluster2')
path3 = os.path.join(os.path.abspath('../../output'), 'cluster3')
path4 = os.path.join(os.path.abspath('../../output'), 'cluster4')

# df import X from part1
df = X

# columns of
col_names = df.columns

# combinations of columns
all_combinations_list_col = [list(itertools.combinations(col_names, r)) for r in range(2, len(col_names))]

# num workers
num_workers = 8

all_combianations_col = [itertools.combinations(col_names, r) for r in range(1, len(col_names))]

all_combinations = list(itertools.chain(*all_combinations_list_col))

# all_combinations[-1]

list(itertools.chain(*all_combinations))
list(itertools.chain(all_combinations))

all_combinations_divide = [a for a in all_combinations if len(a) > 0]


def chunkify(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


all_sub_combinations = chunkify(all_combinations_divide, num_workers)

# get colunms of combinations
df_combibation = df[list(all_sub_combinations[0][0])]
df_combibation_list = [df[list(combination)] for combination in all_sub_combinations[0]]


# df_combibation = all_sub_combinations[0]

class KMeansCluster:
    def __init__(self, df):
        self.df = df
        self.df_combibation = df[list(all_sub_combinations[0][0])]


def get_object_df(df, combinations):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    df_combi_list = []
    scaler_list = []
    labal_list = []
    df_cluster_list = []

    for chunk_combinations in all_sub_combinations:
        # Iterate over each combination in the chunk
        for combinations in chunk_combinations:
            # Accessing the columns in the DataFrame using the combination
            df_combi = df[list(combinations)]
            df_combi_list.append(df_combi)

            x_scaler = MinMaxScaler().fit_transform(df_combi)
            scaled = pd.DataFrame(x_scaler, columns=df_combi.columns)
            scaler_list.append(scaled)
            print(("\n"))
            del scaled, x_scaler

            for i in scaler_list:
                for n in range(2, 5):
                    kmeans = KMeans(n_clusters=n, n_init=10)
                    cluster_labels = kmeans.fit_predict(i)
                    df_cluster_labels = pd.DataFrame(cluster_labels)
                    clusters = silhouette_score(i, kmeans.labels_).__round__(4)

                    labal_list.append([df_combi, df_cluster_labels, clusters, n])
                    df_lables = pd.DataFrame(labal_list)

                    for j, row in df_lables.iterrows():
                        df_concat_col = pd.DataFrame(row[0])
                        df_concat_col['total_time'] = df_original['total_time']
                        df_concat_col['cluster_labels'] = row[1]
                        df_concat_col['score'] = row[2]
                        df_concat_col['clusters'] = row[3]
                        df_cluster_list.append(df_concat_col)
                        print(df_cluster_list)

    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time : {result_time}")

    return df_cluster_list


df_combibation_one_by_one = get_object_df(df, df_combibation)
# print("df_combibation_one_by_one: ", df_combibation_one_by_one)


# Limit to just 120000 rows
list_df_list = df_combibation_one_by_one[:120000]

# multiprocessing part
cpus = 2
parsed_description_split = [[list_df_list[0], list_df_list[1]],
                            [list_df_list[2], list_df_list[3]]
                            ]
print(f"Processing files in directory {list_df_list} using {cpus} CPU cores")
# print("Number of cpus: ", cpus)
print("Number of splits: ", len(parsed_description_split))

with ThreadPoolExecutor(max_workers=cpus) as executor:
    parsed_description_split = list(executor.map(get_object_df, parsed_description_split))
    print(parsed_description_split)
