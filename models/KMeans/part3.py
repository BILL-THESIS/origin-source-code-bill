from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\scaled'
directory_path_lables = 'D:\origin-source-code-bill\models\KMeans\lable'
directory_path_scores = 'D:\origin-source-code-bill\models\KMeans\scores'
pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
print(pkl_files)

scores = []
explode_labels = []
labels = []

for pkl_file in pkl_files:
    file_path = os.path.join(directory_path, pkl_file)
    print("file :::" , file_path)
    variable_name = os.path.splitext(pkl_file)[0]
    print("Var ::" , variable_name)
    df_scaled = pd.read_pickle(file_path)
    print("DF ::" , df_scaled)
    scaled_col = df_scaled.columns

    for n_clusters in range(2,12):
        km = KMeans(n_clusters = n_clusters)
        print("KM :::" ,km)
        km.fit(df_scaled)
        sil_avg = silhouette_score(df_scaled , km.labels_).round(4)
        scores.append([df_scaled.columns.tolist(),sil_avg , n_clusters])
        print("SCORES :::", scores)
        # pd.DataFrame(scores).to_pickle(f"{directory_path_scores}/{df_scaled.columns.tolist()}_scores.pkl")

        cluster_labels = km.fit_predict(df_scaled)
        df_cluster_labels = pd.DataFrame(cluster_labels)
        print("CLUSTER :::" , cluster_labels)

        labels.append([df_scaled.columns.tolist(), n_clusters, sil_avg,cluster_labels])
        x_labels = pd.DataFrame(labels)
        x_labels.to_pickle(f"{directory_path_lables}/labels_final_12.pkl")
        # print("Labels :::", x_labels)
        # x_labels_explode = x_labels.explode(1)
        # print("Explode :::", x_labels_explode)
        print("---------------------------------")
        # explode_labels.append([df_scaled.columns.tolist(), x_labels_explode])
        # print("Explode Labels :::", explode_labels)
        print("---------------------------------")
        # explode_labels_df = pd.DataFrame(explode_labels)
        # df_cluster_labels.to_pickle(f"{directory_path_lables}/{df_scaled.columns.tolist()}_labels.pkl")


        # labels.append([df_scaled.columns.tolist(), n_clusters, sil_avg, x_lables_explode])
        # x_lables.to_pickle(f"{directory_path_lables}/lables_3.pkl")
