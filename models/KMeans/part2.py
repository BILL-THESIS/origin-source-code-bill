from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\combia'
directory_path_scaled = 'D:\origin-source-code-bill\models\KMeans\scaled'

pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
print(pkl_files)

scaler = MinMaxScaler()
scaled_dataframes = []
scores = []
labels = []
result_dfs = []

for csv_file in pkl_files:
    file_path = os.path.join(directory_path, csv_file)
    print("file :::" , file_path)
    variable_name = os.path.splitext(csv_file)[0]
    print("Var ::" , variable_name)
    df_col_combined = pd.read_pickle(file_path)
    print("DF ::" , df_col_combined)

    scaled_data = scaler.fit_transform(df_col_combined)
    print("Scaled :::", scaled_data)
    scaled_df = pd.DataFrame(scaled_data, columns=df_col_combined.columns)
    print("scaled_df_T :::" , scaled_df)
    # scaled_df.to_pickle(f"{directory_path_scaled}/{scaled_df.columns.tolist()}_scaled.pkl")
    # scaled_dataframes.append(scaled_df)

    for n_clusters in range(2,7): #11
        km = KMeans(n_clusters = n_clusters)
        print("KM :::" ,km)
        km.fit(scaled_df)
        sil_avg = silhouette_score(scaled_df , km.labels_).round(4)
        print("SCORES :::", scores)

        cluster_labels = km.fit_predict(scaled_df)
        df_cluster_labels = pd.DataFrame(cluster_labels)
        print("CLUSTER :::" , cluster_labels)

        labels.append([df_col_combined,df_cluster_labels,sil_avg,n_clusters])
        print(labels)
        df_lables = pd.DataFrame(labels)

        for i, row in df_lables.iterrows():
            print("I :::::::::::", i)
            print('\n')

            print("ROW 1 ", row[0])
            print('\n')

            print("ROW 2 ", row[1])
            print('\n')

            print("ROW 3 ", row[2])
            print('\n')

            print("ROW 4 ", row[3])
            print('\n')


            df1 = pd.concat([row[0], row[1]], axis=1)
            print("==========================")

            df1 = df1.rename(columns={0: f'{row[0].columns.to_list()}_{row[3]}'})
            df1['scored'] = row[2]
            df1['clusters'] = row[3]

            print(df1)
            print('\n')

            df1.to_pickle(f'{directory_path_scaled}/{df1.columns[-3]}.pkl')