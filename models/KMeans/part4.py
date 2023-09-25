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

df = pd.read_csv("cluster_lables.csv")