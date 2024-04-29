import pandas as pd
import os
import joblib
import time
import numpy as np


def load_data(file_names):
    df_list = []
    for file_name in file_names:
        results = pd.read_parquet(file_name)
        df_list.append(results)
        print("Data len all:", len(df_list))
    return df_list

def concat_df_list(lists):
    concatenated_df = pd.concat(lists)
    return concatenated_df


def concat_df_list(lists):
    concatenated_df = pd.concat(lists)
    return concatenated_df


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    file_names = (
<<<<<<< HEAD
        'output/q3_c3/q3_c3_2024-03-28 06:00:11.parquet',
        'output/q3_c3/q3_c3_2024-03-28 06:14:04.parquet',
        'output/q3_c3/q3_c3_2024-03-28 06:28:03.parquet',
        'output/q3_c3/q3_c3_2024-03-28 06:42:11.parquet',
        'output/q3_c3/q3_c3_2024-03-28 06:55:51.parquet',
        'output/q3_c3/q3_c3_2024-03-28 07:09:59.parquet',
        'output/q3_c3/q3_c3_2024-03-28 07:24:15.parquet',
        'output/q3_c3/q3_c3_2024-03-28 07:37:25.parquet',
=======
        'output/q3_c3/q3_c3_2024-03-18 13:47:09.parquet',
        'output/q3_c3/q3_c3_2024-03-18 14:23:33.parquet',
        'output/q3_c3/q3_c3_2024-03-19 00:27:43.parquet',
        'output/q3_c3/q3_c3_2024-03-19 00:37:59.parquet',
        'output/q3_c3/q3_c3_2024-03-19 00:48:04.parquet',
        'output/q3_c3/q3_c3_2024-03-19 00:58:20.parquet',
        'output/q3_c3/q3_c3_2024-03-19 01:12:59.parquet',
        'output/q3_c3/q3_c3_2024-03-19 01:23:16.parquet',
>>>>>>> 4c815474 (new version)
    )

    data = load_data(file_names)
    df = concat_df_list(data)
<<<<<<< HEAD
    df_sort_max = df.sort_values(by='Q3', ascending=False).head(10)
    df_sort_min = df.sort_values(by='Q1', ascending=True).head(10)
    print("DF", df_sort_max['Q3'])
    print("========")
    print("DF Min ::", df_sort_min['Q1'])

    # df_sort.to_parquet(f'output/q3_c3/q3_c3_top_10_{time_str}.parquet')
=======
    df_sort = df.sort_values(by='Q3')[::-1].head(10)
    print("DF", df_sort['Q3'])
    df_sort.to_parquet(f'output/q3_c3/q3_c3_top_10_{time_str}.parquet')

>>>>>>> 4c815474 (new version)

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
