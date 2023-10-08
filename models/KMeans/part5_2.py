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

labels = pd.read_pickle('lable/labels_final.pkl')
group_lables = pd.read_pickle('lable/labels_group.pkl')

scaled_dataframes = []

for csv_file in pkl_files:
    file_path = os.path.join(directory_path, csv_file)
    print("file :::" , file_path)
    variable_name = os.path.splitext(csv_file)[0]
    print("Var ::" , variable_name)
    df_col_combined = pd.read_pickle(file_path)
    print("DF ::" , df_col_combined)