from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os



df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

labels = pd.read_pickle('lable/labels_final.pkl')

new_data = {}

for i , row in labels.iterrows():
    col_name = f'{row[0]}_{row[1]}'
    new_data[col_name] = row[3]

df_labels = pd.DataFrame(new_data)
print(df_labels)
# df_labels.to_pickle("labels_group.pkl")