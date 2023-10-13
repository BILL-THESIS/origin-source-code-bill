from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os



df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

directory_path_scores = 'D:\origin-source-code-bill\models\KMeans\scores'
directory_path_labels = 'D:\origin-source-code-bill\models\KMeans\lable'

labels = pd.read_pickle('lable/labels_final.pkl')

pkl_files_scores = [f for f in os.listdir(directory_path_scores) if f.endswith('.pkl')]
pkl_files_labels = [f for f in os.listdir(directory_path_labels) if f.endswith('.pkl')]


for pkl_file_score in pkl_files_scores:
    file_path = os.path.join(directory_path_scores, pkl_file_score)
    print("file :::" , file_path)
    df_score = pd.read_pickle(file_path)
    print("DF Score::" , df_score)

for pkl_files_label in pkl_files_labels:
    file_path = os.path.join(directory_path_labels, pkl_files_label)
    print("file :::" , file_path)
    variable_name = os.path.splitext(pkl_files_label)[0]
    print("Var ::", variable_name)
    var = variable_name.replace('_labels', '')
    df_label = pd.read_pickle(file_path)
    print("DF Label::" , df_label)
    df_label_rename = df_label.rename(columns={ 0 : var})
    print("DF Label Rename::", df_label_rename)
