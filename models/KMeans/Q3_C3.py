import os
import numpy as np
import pandas as pd
import joblib

#
# def calculate_quartiles(data):
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     median = np.median(data)
#     return q1, q3, median
#
#
# def filter_data(data, lower_bound, upper_bound):
#     return data[data.between(lower_bound, upper_bound)]
#
#
# def calculate_coefficient_of_variation(data):
#     mean_value = np.mean(data)
#     standard_deviation = np.std(data)
#     return (standard_deviation / mean_value) * 100
#
#
# def process_cluster(data, cluster_value):
#     cluster_data = data[data['label'] == cluster_value]
#     q1, q3, median = calculate_quartiles(cluster_data['hours'])
#     equal_q3 = filter_data(cluster_data['hours'], q3, np.inf)
#     return q3, equal_q3
#
#
# def process_results(results):
#     cluster_list = []
#     for i in results:
#         df_col_combined = pd.DataFrame(i)
#         df_col_combined['hours'] = pd.to_timedelta(df_col_combined['total_time']).dt.total_seconds() / 3600
#         name_col = df_col_combined.drop(columns=['num', 'score', 'label', 'total_time', 'hours']).columns.to_list()
#
#         quartile_data = {'col': name_col}
#         for cluster_value in df_col_combined['label'].unique():
#             q3, equal_q3 = process_cluster(df_col_combined, cluster_value)
#             quartile_data[f'cluter{cluster_value}_q3'] = [q3]
#             quartile_data[f'shape_c{cluster_value}'] = [equal_q3.shape]
#             quartile_data[f'cv_q3_c{cluster_value}'] = [calculate_coefficient_of_variation(equal_q3['hours'])]
#
#         cluster_list.append(quartile_data)
#
#     return pd.DataFrame(cluster_list)
#
#
# with open('output/results_all3.parquet', 'rb') as f:
#     results = joblib.load(f)
#
# if __name__ == '__main__':
#     percentiles = calculate_quartiles(results[0]['hours'])
#     filter_data(results[0][0]['hours'], percentiles[0], percentiles[1])
#     cv = calculate_coefficient_of_variation(results[0][0]['hours'])
#     c_0 =  process_cluster(results[0][0], 0)
#     results_q = process_results(results)
#     print(results_q)

# results_q = process_results(results)
#
# inport file

with open(f'output/results_all3.parquet', 'rb') as f:
    results = joblib.load(f)

# plt.figure(figsize=(12, 8))
cluster_list = []
quartile_list = []

for i in results:
    df_col_combined = pd.cDataFrame(i)
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
        'Q3' : [(abs(cluter0_q3 - cluter1_q3) + abs(cluter0_q3 - cluter2_q3) + abs(cluter2_q3 - cluter1_q3)) / 3 ],
        'shape_c0': [c_0.shape],
        'cv_q3_c0': [coefficient_of_variation(equal_q3_c0['hours'])],
        'shape_c1': [c_1.shape],
        'cv_q3_c1': [coefficient_of_variation(equal_q3_c1['hours'])],
        'shape_c2': [c_2.shape],
        'cv_q3_c2': [coefficient_of_variation(equal_q3_c2['hours'])]
    }

    cluster_list.append(quartile_data)
    results_q_c3 = pd.DataFrame(cluster_list)

    results_q_c3['Q3'].sort_values

    # top1_c3_q3 = results_q_c3.iloc[2]



