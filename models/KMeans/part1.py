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
                                        'commits', 'additions', 'deletions', 'changed_files'
                                        ])]

X['begin_D'] = pd.to_numeric(X['begin_D'] , errors='coerce')
X['begin_B'] = pd.to_numeric(X['begin_B'] , errors='coerce')
X['begin_CP'] = pd.to_numeric(X['begin_CP'] , errors='coerce')
X['begin_C'] = pd.to_numeric(X['begin_C'] , errors='coerce')
X['begin_OOA'] = pd.to_numeric(X['begin_OOA'] , errors='coerce')
X['end_D'] = pd.to_numeric(X['end_D'] , errors='coerce')
X['end_B'] = pd.to_numeric(X['end_B'] , errors='coerce')
X['end_CP'] = pd.to_numeric(X['end_CP'] , errors='coerce')
X['end_C'] = pd.to_numeric(X['end_C'] , errors='coerce')
X['end_OOA'] = pd.to_numeric(X['end_OOA'] , errors='coerce')

X['D_change'] = X['begin_D'] - X['end_D']
X['B_change'] = X['begin_B'] - X['end_B']
X['CP_change'] = X['begin_CP'] - X['end_CP']
X['C_change'] = X['begin_C'] - X['end_C']
X['OOA_change'] = X['begin_OOA'] - X['end_OOA']

X['D_percent'] = (X['D_change'] / X['begin_D']) * 100
X['B_percent'] = (X['B_change'] / X['begin_B']) * 100
X['CP_percent'] = (X['CP_change'] / X['begin_CP']) * 100
X['C_percent'] = (X['C_change'] / X['begin_C']) * 100
X['OOA_percent'] = (X['OOA_change'] / X['begin_OOA']) * 100

column = X.columns

directory_path_combia = '../../models/KMeans/combia2'

start_time = time.time()
start_time_gmt = time.gmtime(start_time)
start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)


# for r in range(1, len(X.columns) + 1):
#     # uesd itertools import combinations to calculater
#     for column_combination in combinations(X.columns, r):
#         combined_df = X[list(column_combination)]
#         print(combined_df.columns)
#         combined_df.to_parquet(f"{directory_path_combia}/{combined_df.columns.tolist()}.parquet")


end_time = time.time()
result_time = end_time - start_time
result_time_gmt = time.gmtime(result_time)
result_time = time.strftime("%H:%M:%S", result_time_gmt)
print(f"Total time: {result_time}")
