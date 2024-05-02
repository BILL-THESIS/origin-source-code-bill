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

if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    file_names = (
        'output/q3_c2/q3_c2_2024-03-18 12:05:27.parquet',
        'output/q3_c2/q3_c2_2024-03-18 12:17:26.parquet',
        'output/q3_c2/q3_c2_2024-03-18 12:25:47.parquet',
        'output/q3_c2/q3_c2_2024-03-18 12:33:59.parquet',
        'output/q3_c2/q3_c2_2024-03-18 12:50:36.parquet',
        'output/q3_c2/q3_c2_2024-03-18 12:42:26.parquet',
        'output/q3_c2/q3_c2_2024-03-18 13:12:53.parquet',
        'output/q3_c2/q3_c2_2024-03-18 13:12:53.parquet',
    )

    data = load_data(file_names)
    # print("DATA :", data)


    df = concat_df_list(data)
    df_sort = df.sort_values(by='Q3')[::-1].head(10)
    print("DF", df_sort)
    # df_sort.to_parquet(f'output/q3_c2/q3_c2_top_10{time_str}.parquet.gz', compression='gzip')

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
