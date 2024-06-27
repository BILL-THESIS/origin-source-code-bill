import pandas as pd
import os
import joblib
import time
import numpy as np


def load_data(file_names):
    for file_name in file_names:
        with open(os.path.join(file_name), 'rb') as f:
            results = joblib.load(f)
    return results


def process_cluster(cluster_data):
    hours = cluster_data['hours']
    q1 = np.percentile(hours, 25)
    q3 = np.percentile(hours, 75)
    median = np.median(hours)
    equal_q3 = cluster_data[cluster_data['hours'] >= q3]
    return q1, q3, median, equal_q3


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
            q1, q3, median, equal_q3 = process_cluster(cluster_data)
            quartile_data[f'cluter{cluster_value}_q3'] = [q3]
            quartile_data[f'shape_c{cluster_value}'] = [equal_q3.shape]
            quartile_data[f'cv_q3_c{cluster_value}'] = [coefficient_of_variation(equal_q3['hours'])]
<<<<<<< HEAD
            quartile_data[f'label'] = [df_col_combined['label'].tolist()]
=======
>>>>>>> 4c815474 (new version)
        cluster_list.append(quartile_data)

    return cluster_list


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    file_names = (
        'output/2/results_all2_2024-03-17 15:06:58.parquet.gz',
        # 'output/2/results_all2_2024-03-17 15:14:06.parquet.gz',
        # 'output/2/results_all2_2024-03-17 15:25:48.parquet.gz',
        # 'output/2/results_all2_2024-03-17 15:32:41.parquet.gz',
        # 'output/2/results_all2_2024-03-17 15:40:12.parquet.gz',
        # 'output/2/results_all2_2024-03-17 15:50:00.parquet.gz',
        # 'output/2/results_all2_2024-03-17 16:02:37.parquet.gz',
        # 'output/2/results_all2_2024-03-17 16:10:28.parquet.gz',
    )

    results = load_data(file_names)
    data_quartile = qurtile_data(results)
    df = pd.DataFrame(data_quartile)
    print("df q3 :: ", df.columns.to_list())

    df['Q3'] = df.apply(lambda row: (abs(row['cluter0_q3'][0] - row['cluter1_q3'][0]) + abs(
        row['cluter1_q3'][0] - row['cluter0_q3'][0])) / 3, axis=1)

    df_sort = df.sort_values(by='Q3')[::-1].head(10)
    print("normal df:: ", df.sort_values(by='Q3').head(10))
    print("top 10 ::", df_sort)

    df_sort.to_parquet(f'output/q3_c2/q3_c2_{time_str}.parquet')

    # with open(f'output/2/results_all2_{time_str}.parquet.gz', 'wb') as f:
    #     joblib.dump(results_all2, f, compress=('gzip'))
    #     print("svae file cluster 2  Done!!")

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
<<<<<<< HEAD
    print("Total time {:.2f} hours".format(time_hours))
=======
    print("Total time {:.2f} hours".format(time_hours))
>>>>>>> 4c815474 (new version)
