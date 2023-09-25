from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\scaled'
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
print(csv_files)

scores = []
list_scores = []

for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    print("file :::" , file_path)
    variable_name = os.path.splitext(csv_file)[0]
    print("Var ::" , variable_name)
    df_scaled = pd.read_csv(file_path)
    print("DF ::" , df_scaled)
    scaled_col = df_scaled.columns

    for n_clusters , scaled_col in range(2,10):
        km = KMeans(n_clusters = n_clusters)
        print("KM :::" ,km)
        km.fit(df_scaled)
        sil_avg = silhouette_score(df_scaled , km.labels_).round(4)
        scores.append([sil_avg , n_clusters])