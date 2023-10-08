from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\group_time'
directory_path_time = 'D:\origin-source-code-bill\models\KMeans\group_time'
df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
group_lables = pd.read_pickle('lable/List_df_labels.pkl')

pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
print(pkl_files)

results = []
cluster_list = []

for pkl_file in pkl_files:
    file_path = os.path.join(directory_path, pkl_file)
    # print("file :::" , file_path)
    variable_name = os.path.splitext(pkl_file)[0]
    # print("Var ::" , variable_name)
    df_col_combined = pd.read_pickle(file_path)
    # print("DF ::" , df_col_combined)
    print('\n')

    df_last_col = df_col_combined.iloc[:,-1].unique()
    print("Last Columns ::" , df_last_col)
    print("\n")

    col_srt = df_col_combined.columns[-1]

    for cluster_value in df_last_col :
        print("I :::", cluster_value)
        cluster_data = df_col_combined[df_col_combined[col_srt] == cluster_value]
        cluster_list.append(cluster_data)
        cluster_data_df = pd.DataFrame(cluster_data)
        print(f"Cluster {cluster_value}:")
        print("Minimum timedelta:", cluster_data['total_time'].min())
        print("Average timedelta:", cluster_data['total_time'].mean())
        print("Maximum timedelta:", cluster_data['total_time'].max())
        print("Cluster shape:", cluster_data.shape)
        print("\n")

        results_dict = {
            'name_col' : col_srt ,
            'cluster' : cluster_value ,
            f'Minimum{cluster_value}' : cluster_data['total_time'].min() ,
            f'Average{cluster_value}' : cluster_data['total_time'].mean(),
            f'Maximum{cluster_value}': cluster_data['total_time'].max(),
        }

    results.append(results_dict)
    results_df = pd.DataFrame(results)