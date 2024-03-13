import os
import numpy as np
import pandas as pd
import joblib

with open(f'output/results_all4.parquet', 'rb') as f:
    results = joblib.load(f)

# plt.figure(figsize=(12, 8))
cluster_list = []
quartile_list = []

for i in results:
    df_col_combined = pd.DataFrame(i)
    print(df_col_combined.columns)

    date = pd.to_timedelta(df_col_combined['total_time'])
    time_hours = date.dt.total_seconds() / 3600

    df_col_combined['hours'] = time_hours
    # print("DF ::" , df_col_combined.columns)

    df_last_col = df_col_combined['label'].unique()
    col_srt = df_col_combined.columns[-3]
    name_col = df_col_combined.drop(columns=['num', 'score', 'label', 'total_time', 'hours'])
    name_col = name_col.columns.to_list()

    for cluster_value in df_last_col:
        print("I :::", cluster_value)
        cluster_data = df_col_combined[df_col_combined[col_srt] == cluster_value]
        print("================")

        if cluster_value == 0:
            c_0 = cluster_data['hours']
            cluter0_q1 = np.percentile(c_0, 25)
            cluter0_q3 = np.percentile(c_0, 75)
            cluter0_median = np.median(c_0)

            less_q1_c0 = cluster_data[cluster_data['hours'] < cluter0_q1]
            more_q3_c0 = cluster_data[cluster_data['hours'] > cluter0_q3]
            equal_q3_c0 = cluster_data[cluster_data['hours'] >= cluter0_q3]

        if cluster_value == 1:
            c_1 = cluster_data['hours']
            cluter1_q1 = np.percentile(cluster_data['hours'], 25)
            cluter1_q3 = np.percentile(cluster_data['hours'], 75)
            cluter1_median = np.median(cluster_data['hours'])

            less_q1_c1 = cluster_data[cluster_data['hours'] < cluter1_q1]
            more_q3_c1 = cluster_data[cluster_data['hours'] > cluter1_q3]
            equal_q3_c1 = cluster_data[cluster_data['hours'] >= cluter1_q3]

        if cluster_value == 2:
            c_2 = cluster_data['hours']
            cluter2_q1 = np.percentile(cluster_data['hours'], 25)
            cluter2_q3 = np.percentile(cluster_data['hours'], 75)
            cluter2_median = np.median(cluster_data['hours'])

            less_q1_c2 = cluster_data[cluster_data['hours'] < cluter2_q1]
            more_q3_c2 = cluster_data[cluster_data['hours'] > cluter2_q3]
            equal_q3_c2 = cluster_data[cluster_data['hours'] >= cluter2_q3]

        if cluster_value == 3:
            c_3 = cluster_data['hours']
            cluter3_q1 = np.percentile(cluster_data['hours'], 25)
            cluter3_q3 = np.percentile(cluster_data['hours'], 75)
            cluter3_median = np.median(cluster_data['hours'])

            less_q1_c3 = cluster_data[cluster_data['hours'] < cluter3_q1]
            more_q3_c3 = cluster_data[cluster_data['hours'] > cluter3_q3]
            equal_q3_c3 = cluster_data[cluster_data['hours'] >= cluter3_q3]


    def coefficient_of_variation(data):
        mean_value = np.mean(data)
        standard_deviation = np.std(data)
        coefficient_variation = (standard_deviation / mean_value) * 100
        return coefficient_variation


    quartile_data = {
        'col': name_col,
        'cluter0_q3': [cluter0_q3],
        'cluter1_q3': [cluter1_q3],
        'cluter2_q3': [cluter2_q3],
        'cluter3_q3': [cluter3_q3],
        'Q3': [(abs(cluter0_q3 - cluter1_q3) + abs(cluter0_q3 - cluter2_q3) + abs(cluter0_q3 - cluter3_q3) +
                abs(cluter2_q3 - cluter1_q3) + abs(cluter2_q3 - cluter3_q3) + abs(cluter3_q3 - cluter1_q3)) / 4],
        'shape_c0': [c_0.shape],
        'cv_q3_c0': [coefficient_of_variation(equal_q3_c0['hours'])],
        'shape_c1': [c_1.shape],
        'cv_q3_c1': [coefficient_of_variation(equal_q3_c1['hours'])],
        'shape_c2': [c_2.shape],
        'cv_q3_c2': [coefficient_of_variation(equal_q3_c2['hours'])],
        'shape_c3': [c_3.shape],
        'cv_q3_c3': [coefficient_of_variation(equal_q3_c3['hours'])]
    }

    cluster_list.append(quartile_data)
    results_q_c4 = pd.DataFrame(cluster_list)
    print(results_q_c4['Q3'].sort_values)

    # top1_c4_q3 = results_q_c4.iloc[7]