from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans'
directory_path_scaled = 'D:\origin-source-code-bill\models\KMeans\scaled'
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
print(csv_files)

scaler = MinMaxScaler()
scaled_dataframes = []

for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    print("file :::" , file_path)
    variable_name = os.path.splitext(csv_file)[0]
    print("Var ::" , variable_name)
    df_col_combined = pd.read_csv(file_path)
    print("DF ::" , df_col_combined)
    scaled_data = scaler.fit_transform(df_col_combined)
    print("Scaled :::", scaled_data)
    scaled_df = pd.DataFrame(scaled_data, columns=df_col_combined.columns)
    # scaled_df = scaled_df.T
    # scaled_drop_col =  scaled_df.columns
    #scaled_drop_col = [i for i in range(len(scaled_df.columns))]
    # scaled_df.reset_index(drop=True, inplace=True)
    print("scaled_df_T :::" , scaled_df)
    scaled_df.to_csv(f"{directory_path_scaled}/{scaled_df.columns.tolist()}_scaled.csv", index=False)
    scaled_dataframes.append(scaled_df)
