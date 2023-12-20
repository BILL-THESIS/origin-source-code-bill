import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

from sqlalchemy.dialects.mssql.information_schema import columns

df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

X = df_original
X = X.rename(columns={"begin_Dispensables": "begin_D",
                      "begin_Bloaters": "begin_B",
                      "begin_Change Preventers": "begin_CP",
                      "begin_Couplers": "begin_C",
                      "begin_Object-Orientation Abusers": "begin_OOA",
                      "end_Dispensables": "end_D",
                      "end_Bloaters": "end_B",
                      "end_Change Preventers": "end_CP",
                      "end_Couplers": "end_C",
                      "end_Object-Orientation Abusers": "end_OOA",
                      })

X = X.loc[:, ~df_original.columns.isin(['begin_sha', 'end_sha', 'begin_time', 'end_time',
                                        'total_time', 'time',
                                        'commits', 'additions', 'deletions', 'changed_files',
                                        'end_D', "end_B", "end_CP", "end_C", "end_OOA"
                                        ])]
# X.iloc[:, [4,5,6,7,8]]


column = X.columns

directory_path_combia = '../../models/KMeans/combia2'

start_time = time.time()
start_time_gmt = time.gmtime(start_time)
start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)


for r in range(1, len(X.columns) + 1):
    # uesd itertools import combinations to calculater
    for column_combination in combinations(X.columns, r):
        combined_df = X[list(column_combination)]
        print(combined_df.columns)
        combined_df.to_parquet(f"{directory_path_combia}/{combined_df.columns.tolist()}.parquet")


end_time = time.time()
result_time = end_time - start_time
result_time_gmt = time.gmtime(result_time)
result_time = time.strftime("%H:%M:%S", result_time_gmt)
print(f"Total time: {result_time}")
