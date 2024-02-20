import time
import pandas as pd
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

class KMeansCluster:
    def __init__(self, df_original_20_col: pd.DataFrame, scarler):
        self.df = df_original_20_col
        self.scarler = scarler
        self.all_combinations = [a for a in itertools.combinations(self.df.columns, r) for r in range(2, len(self.df.columns) + 1) if len(a) > 0]

    def fit_scaler(self, df):
        scaled = self.scarler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns)

    def kmeans_cluster(self, scaled_df):
        return [[silhouette_score(scaled_df, KMeans(n_clusters=n, n_init=10).fit_transform(scaled_df).labels_), n] for n in range(2, 5)]

if __name__ == '__main__':
    start = time.time()

    minmax_scaler = MinMaxScaler()
    df_original_20_col = pd.read_parquet('seatunnal_20col.parquet')

    bill = KMeansCluster(df_original_20_col, minmax_scaler)

    sub_c2 = bill.all_combinations[-10:]

    asd = [bill.fit_scaler(df_original_20_col[list(i[0])]) for i in sub_c2]



    print("Total time {:.2f} hours".format((time.time() - start) / 3600))