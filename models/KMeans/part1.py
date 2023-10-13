from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
X = df_original.loc[:, ~df_original.columns.isin(['begin_sha', 'end_sha', 'begin_time','end_time',
                                'total_time',
                                'time'
                                ])]

X.iloc[:, [4,5,6,7,8]]

df = pd.DataFrame(X.iloc[:, [4,5,6,7,8]])
columns = df.columns

directory_path_combia = 'D:\origin-source-code-bill\models\KMeans\combia'

for r in range(1, len(columns) + 1):
    for column_combination in combinations(columns , r):
        combined_df = df[list(column_combination)]
        print(combined_df)
        # combined_df.to_pickle(f"{directory_path_combia}/{combined_df.columns.tolist()}.pkl")