import pandas as pd
import os
import joblib
import time
import gzip
from datetime import datetime
import numpy as np


def load_data(file_names):
    df_list = []
    for file_name in file_names:
        with open(os.path.join(file_name), 'rb') as f:
            results = joblib.load(f)
            df_list.append(results)
            print("Data len all:", len(df_list))
    return df_list


def process_dataframe(data_list):
    concatenated_list = []
    for sub_list in data_list:
        # print("I", sub_list)
        for sub_in_sub in sub_list:
            col = sub_in_sub['df']
            cluster_2 = sub_in_sub['2']
            cluster_3 = sub_in_sub['3']
            cluster_4 = sub_in_sub['4']
            concatenated_list.append([col, cluster_2, cluster_3, cluster_4])

        df = pd.DataFrame(concatenated_list)

        df2 = pd.DataFrame(df[1].to_list(), columns=['num', 'score', 'lable'])
        df3 = pd.DataFrame(df[2].to_list(), columns=['num', 'score', 'lable'])
        df4 = pd.DataFrame(df[3].to_list(), columns=['num', 'score', 'lable'])

        # print("data frame 2", df2, "\n")
        # print("data frame 3", df3, "\n")
        # print("data frame 4", df4, "\n")

        new_df2 = pd.concat([df[0], df2], axis=1)
        new_df3 = pd.concat([df[0], df3], axis=1)
        new_df4 = pd.concat([df[0], df4], axis=1)
    # return new_df2
    return new_df2, new_df3, new_df4


def process_data_cluster(df_cluster, df_20_col, df_original):
    label_lists = []
    for i, row in df_cluster.iterrows():
        print("I", i)
        # print("Row 0", row[0]) # row 0 is columns name of table combinations
        # print("Row 1 N", row['num'])
        # print("Row 2 score", row['score'])
        # print("Row 3 lable", row['lable'])
        selected_cols = [col for col in row[0] if col is not None]
        data_list = pd.DataFrame({
            f'num': row['num'],
            f'score': row['score'],
            f'label': row['lable']
        })
        df_concat = pd.concat([df_20_col[selected_cols], data_list, df_original['total_time']], axis=1)
        label_lists.append(df_concat)
    return label_lists


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    df_original = pd.read_parquet('../../../Sonar/seatunnel_all_information.parquet')
    df_20_col = pd.read_parquet('../../output/seatunnal_20col.parquet')

    file_names = (
        # '../../output/1709313625.730204.pkl',
        # '../../output/1709313723.167009.pkl',
        # '../../output/1709313756.702853.pkl',
        # '../../output/1709313838.753286.pkl',
        # '../../output/1709313844.533946.pkl',
        # '../../output/1709313907.672706.pkl',
        # '../../output/1709314037.231783.pkl',
        '../../output/1709327621.503408.pkl',
    )

    df_lsit = load_data(file_names)

    # new_df2 = process_dataframe(df_lsit)
    # new_df3 = process_dataframe(df_lsit)
    # new_df4 = process_dataframe(df_lsit)
    new_df2, new_df3, new_df4 = process_dataframe(df_lsit)

    # print("Cluster2 :", new_df2)
    # print("Cluster3 :", new_df3)
    # print("Cluster4 :", new_df4)

    # del df_lsit

    results_all2 = process_data_cluster(new_df2, df_20_col, df_original)
    results_all3 = process_data_cluster(new_df3, df_20_col, df_original)
    results_all4 = process_data_cluster(new_df4, df_20_col, df_original)

    with open(f'output/2/results_all2_{time_str}.parquet.gz', 'wb') as f:
        joblib.dump(results_all2, f, compress=('gzip'))
        print("svae file cluster 2  Done!!")

    with open(f'output/3/results_all3_{time_str}.parquet.gz', 'wb') as f:
        joblib.dump(results_all3, f, compress=('gzip'))
        print("svae file cluster 3  Done!!")
    # #
    with open(f'output/4/results_all4_{time_str}.parquet.gz', 'wb') as f:
        joblib.dump(results_all4,f, compress=('gzip'))
        print("svae file cluster 4  Done!!")

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))