import numpy as np
import pandas as pd
import itertools

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

class KMeansCluster:

    def __init__(self, df):
        self.df = df

    def chunkify(self, lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def all_sub_combinations(self, combinations):
        return self.chunkify(combinations, 4)

    def scale_data(self, df):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns)

    def perform_kmeans(self, data):
        kmeans_results = []
        for n_clusters in range(2, 5):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(data)
            silhouette_score_value = silhouette_score(data, kmeans.labels_)
            kmeans_results.append([silhouette_score_value, kmeans.labels_, n_clusters])
        return kmeans_results

if __name__ == '__main__':
    df_X = pd.read_parquet('seatunnal_20col.parquet')
    col_names = df_X.columns
    all_combinations = [list(comb) for r in range(2, len(col_names)) for comb in itertools.combinations(col_names, r) if len(comb) > 0]

    kmeans_cluster = KMeansCluster(df_X)

    all_sub_combinations = kmeans_cluster.all_sub_combinations(all_combinations)

    scalers = [kmeans_cluster.scale_data(df_X[list(combination)]) for combination in all_sub_combinations[0]]

    kmeans_results = [kmeans_cluster.perform_kmeans(scaled_data) for scaled_data in scalers]

    df_result = all_sub_combinations
