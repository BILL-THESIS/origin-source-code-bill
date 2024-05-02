import pandas as pd
import os
import joblib
import time
import numpy as np


# %%
def loop_check_value_q(df):
    index_list = []
    min_values_list = []
    max_values_list = []

    for i in range(len(df)):

        x = df.iloc[i].sort_values()
        # print('index:', df.index[i])
        print('value:', x)

        min = x[0]
        max = x[2]
        # print('min:', x[0])
        # print('max:', x[2])

        index_list.append(df.index[i])
        min_values_list.append(min)
        max_values_list.append(max)

    return index_list, min_values_list, max_values_list


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    # directory_path_max = r'../../models/KMeans/output/q3_c3/q3_c3_top_10_outliers2024-04-29 04:19:07.parquet'
    directory_path_max = r'../output/q3_c3/q3_c3_top_10_normal2024-04-30 04:57:39.parquet'
    df_20_col = pd.read_parquet('../output/seatunnal_20col.parquet')
    df_original = pd.read_parquet('../../../Sonar/seatunnel_all_information.parquet')

    df_max = pd.read_parquet(directory_path_max)
    print(df_max.col)

    df_max_sort = df_max[['col', 'cluter0_q3', 'cluter1_q3', 'cluter2_q3',
                          'cluter0_q1', 'cluter1_q1', 'cluter2_q1',
                          'label']]

    q3 = df_max_sort.iloc[:, [1, 2, 3]]
    q1 = df_max_sort.iloc[:, [4, 5, 6]]

    index, min_q3, max_q3 = loop_check_value_q(q3)
    index, min_q1, max_q1 = loop_check_value_q(q1)

    # %%
    concat_col = pd.concat([pd.DataFrame(index, columns=['index']), pd.DataFrame(min_q1, columns=['min']),
                            pd.DataFrame(max_q3, columns=['max'])], axis=1)

    concat_col = pd.DataFrame(concat_col)
    concat_col.set_index('index', inplace=True)

    df_max_class = pd.merge(df_max_sort, concat_col, left_index=True, right_index=True)
    # df_max_class.to_parquet('../../models/KMeans/output/q3_c3/df_max_time_class_normal.parquet')
    # df_max_class.to_parquet('../../models/KMeans/output/q3_c3/df_max_time_class_outliers.parquet')

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
