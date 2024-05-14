import pandas as pd
import os
import joblib
import time
import numpy as np


def load_data(file_names):
    with open(os.path.join(file_names), 'rb') as f:
        results = joblib.load(f)
    return results


def process_cluster(cluster_data):
    hours = cluster_data['hours']
    q1 = np.percentile(hours, 25)
    q3 = np.percentile(hours, 75)
    IQR = q3 - q1
    ul = q3 + 1.5 * IQR
    ll = q1 - 1.5 * IQR
    outliers = hours[(hours < ll) | (hours > ul)]
    median = np.median(hours)
    equal_q3 = cluster_data[cluster_data['hours'] >= q3]
    equal_q1 = cluster_data[cluster_data['hours'] <= q1]
    return q1, q3, median, equal_q3, equal_q1, outliers, IQR, ul, ll


def coefficient_of_variation(data):
    mean_value = np.mean(data)
    standard_deviation = np.std(data)
    coefficient_variation = (standard_deviation / mean_value) * 100
    return coefficient_variation


def qurtile_data(cluster_data):
    cluster_list = []
    for i in cluster_data:
        df_col_combined = pd.DataFrame(i)
        df_col_combined['hours'] = pd.to_timedelta(df_col_combined['total_time']).dt.total_seconds() / 3600
        name_col = df_col_combined.drop(columns=['num', 'score', 'label', 'total_time', 'hours']).columns.to_list()

        quartile_data = {'col': name_col}
        # clustr values is cluster number 0, 1, 2
        for cluster_value in df_col_combined['label'].unique():
            cluster_data = df_col_combined[df_col_combined['label'] == cluster_value]
            q1, q3, median, equal_q3, equal_q1, outliers, IQR, ul, ll = process_cluster(cluster_data)
            quartile_data[f'cluter{cluster_value}_q3'] = [q3]
            quartile_data[f'cluter{cluster_value}_q1'] = [q1]
            quartile_data['outliers'] = [outliers]
            quartile_data['IQR'] = [IQR]
            quartile_data['ul'] = [ul]
            quartile_data['ll'] = [ll]
            quartile_data[f'outliers_c{cluster_value}'] = [outliers.shape]
            quartile_data[f'shape_c{cluster_value}_q3'] = [equal_q3.shape]
            quartile_data[f'shape_c{cluster_value}_q1'] = [equal_q1.shape]
            quartile_data[f'cv_q3_c{cluster_value}'] = [coefficient_of_variation(equal_q3['hours'])]
            quartile_data[f'cv_q1_c{cluster_value}'] = [coefficient_of_variation(equal_q1['hours'])]
            quartile_data[f'label'] = df_col_combined['label'].tolist()
            quartile_data[f'median'] = median
            quartile_data[f'mean_q3'] = q3.mean()
        cluster_list.append(quartile_data)

    return cluster_list


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    file_names = (
        '/home/bill/output/3/results_all3_2024-03-17 13:50:14.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:05:40.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:12:44.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:19:27.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:27:06.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:34:44.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:48:36.parquet.gz'
        # '/home/bill/output/3/results_all3_2024-03-17 14:58:55.parquet.gz'
    )

    results = load_data(file_names)

    data_quartile = qurtile_data(results)
    df = pd.DataFrame(data_quartile)
    print("df q3 :: ", df.columns.to_list())

    df['Q3'] = df.apply(lambda row: (abs(row['cluter0_q3'][0] - row['cluter1_q3'][0]) +
                                     abs(row['cluter0_q3'][0] - row['cluter2_q3'][0]) +
                                     abs(row['cluter1_q3'][0] - row['cluter0_q3'][0]) +
                                     abs(row['cluter1_q3'][0] - row['cluter2_q3'][0]) +
                                     abs(row['cluter2_q3'][0] - row['cluter0_q3'][0]) +
                                     abs(row['cluter2_q3'][0] - row['cluter1_q3'][0])
                                     ) / 3, axis=1)

    df['Q1'] = df.apply(lambda row: (abs(row['cluter0_q1'][0] - row['cluter1_q1'][0]) +
                                     abs(row['cluter0_q1'][0] - row['cluter2_q1'][0]) +
                                     abs(row['cluter1_q1'][0] - row['cluter0_q1'][0]) +
                                     abs(row['cluter1_q1'][0] - row['cluter2_q1'][0]) +
                                     abs(row['cluter2_q1'][0] - row['cluter0_q1'][0]) +
                                     abs(row['cluter2_q1'][0] - row['cluter1_q1'][0])
                                     ) / 3, axis=1)

    df['max_q3_equa'] = df['Q3'].max()
    df['min_q3_equa'] = df['Q3'].min()
    df['max_q1_equa'] = df['Q1'].max()
    df['min_q1_equa'] = df['Q1'].min()

    # df_sort = df.sort_values(by='Q3', ascending=True)

    df.to_parquet(f'output/q3_c3/q3_c3_otliers{time_str}.parquet')

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
