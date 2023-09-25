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

X.iloc[:, [0,1,2]]

df = pd.DataFrame(X.iloc[:, [0,1,2]])
columns = df.columns

for r in range(1, len(columns) + 1):
    for column_combination in combinations(columns , r):
        combined_df = df[list(column_combination)]
        print(combined_df)
        combined_df.to_csv(f"{combined_df.columns.tolist()}.csv", index=False)