import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)


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

    data_fit = fit_scaler(X)

    X_fit = data_fit[['commits', 'additions', 'deletions',
                      'changed_files',
                      'created_Bloaters', 'created_Change Preventers',
                      'created_Couplers', 'created_Dispensables',
                      'created_Object-Orientation Abusers', 'created_Uncategorized',
                      'ended_Bloaters', 'ended_Change Preventers',
                      'ended_Couplers', 'ended_Dispensables',
                      'ended_Object-Orientation Abusers', 'ended_Uncategorized']]
