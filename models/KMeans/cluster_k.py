import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def fit_scaler(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)


def fit_kmeans(df_scalar):

    results = []
    for n in range(2, 4):
        kmeans = KMeans(n_clusters=n, n_init=10)
        kmeans.fit(df_scalar)
        labels = kmeans.predict(df_scalar)
        labels_cluster = kmeans.labels_
        print(f"Silhouette score for {n} clusters: {silhouette_score(df_scalar, labels)}")
        results.append((n, silhouette_score(df_scalar, labels), labels_cluster))
    return results


if __name__ == "__main__":
    # Load the data
    data = pd.read_parquet("../output/ozone_prepare_to_train.parquet")

    X = data[['commits', 'additions', 'deletions',
              'changed_files',
              'created_Bloaters', 'created_Change Preventers',
              'created_Couplers', 'created_Dispensables',
              'created_Object-Orientation Abusers', 'created_Uncategorized',
              'ended_Bloaters', 'ended_Change Preventers',
              'ended_Couplers', 'ended_Dispensables',
              'ended_Object-Orientation Abusers', 'ended_Uncategorized']]
    y = data['total_time_hours']

    df_list = []

    data_fit = fit_scaler(X)
    results_Kmean = fit_kmeans(data_fit)
    df_Kmean = pd.DataFrame(results_Kmean, columns=['n_clusters', 'silhouette_score', 'labels'])

