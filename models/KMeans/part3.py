from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\scaled'
df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

path2 = 'D:\origin-source-code-bill\models\KMeans\cluster2'
path3 = 'D:\origin-source-code-bill\models\KMeans\cluster3'
path4 = 'D:\origin-source-code-bill\models\KMeans\cluster4'
path5 = 'D:\origin-source-code-bill\models\KMeans\cluster5'

pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
print(pkl_files)

for pkl_file in pkl_files:
    file_path = os.path.join(directory_path, pkl_file)
    # print("file :::" , file_path)
    variable_name = os.path.splitext(pkl_file)[0]
    print("Var ::" , variable_name)
    df_col_combined = pd.read_pickle(file_path)
    print("DF ::" , df_col_combined['clusters'].values[0])
    print('\n')

    if df_col_combined['clusters'].values[0] == 2:
        merged_df3 = pd.concat([df_original['total_time'], df_col_combined] , axis=1).reindex(df_col_combined.index)
        merged_df3.to_pickle(f'{path2}/{df_col_combined.columns[-3]}.pkl')

    if df_col_combined['clusters'].values[0] == 3:
        merged_df2 = pd.concat([df_original['total_time'], df_col_combined] , axis=1).reindex(df_col_combined.index)
        merged_df2.to_pickle(f'{path3}/{df_col_combined.columns[-3]}.pkl')

    if df_col_combined['clusters'].values[0] == 4:
        merged_df3 = pd.concat([df_original['total_time'], df_col_combined] , axis=1).reindex(df_col_combined.index)
        merged_df3.to_pickle(f'{path4}/{df_col_combined.columns[-3]}.pkl')

    if df_col_combined['clusters'].values[0] == 5:
        merged_df3 = pd.concat([df_original['total_time'], df_col_combined] , axis=1).reindex(df_col_combined.index)
        merged_df3.to_pickle(f'{path5}/{df_col_combined.columns[-3]}.pkl')
    else:
        print("non")