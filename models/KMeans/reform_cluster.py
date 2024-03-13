# import joblib
# import pandas as pd
#
# try:
#     with open(f'/Users/bill/origin-source-code-bill/models/KMeans/output/kmeans_0.csv', 'rb') as f:
#         results = joblib.load(f)
#     # Continue with your data processing
# except Exception as e:
#     print("An error occurred while reading the Parquet file:", e)
#
# # read_file output afterthen we can use the data for further processing
# df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
# df_20_col = pd.read_parquet('output/seatunnal_20col.parquet')
#
# df_20_col_time = pd.concat([df_20_col, df_original['total_time']], axis=1)
#
# df = pd.DataFrame(results)
#
# df2 = pd.DataFrame(df['2'].to_list(), columns=['num_2', 'score_2', 'lable_2'])
# df3 = pd.DataFrame(df['3'].to_list(), columns=['num_3', 'score_3', 'lable_3'])
# df4 = pd.DataFrame(df['4'].to_list(), columns=['num_4', 'score_4', 'lable_4'])
#
# # Concatenate the original DataFrame (excluding column nuber cluster) with the transposed DataFrame
#
# new_df2 = pd.concat([df['df'], df2], axis=1)
# new_df3 = pd.concat([df['df'], df3], axis=1)
# new_df4 = pd.concat([df['df'], df4], axis=1)
#


# def process_dataframe(new_df2, df_20_col, df_original):
#     label_lists = []
#     for i, row in new_df2.iterrows():
#         print("I", i)
#         print("Row 0", row[0])
#         print("Row 1", row[1])
#         print("Row 2", row[2])
#         print("Row3 ", row[3])
#         selected_cols = [col for col in row[0] if col is not None]
#         data_list = pd.DataFrame({
#             'num': row[1],
#             'score': row[2],
#             'label': row[3]
#         })
#         df_concat = pd.concat([df_20_col[selected_cols], data_list, df_original['total_time']], axis=1)
#         label_lists.append(df_concat)
#     return label_lists


# results_all2 = process_dataframe(new_df2, df_20_col, df_original)
# results_all3 = process_dataframe(new_df3, df_20_col, df_original)
# results_all4 = process_dataframe(new_df4, df_20_col, df_original)
#
# with open('output/results_all4.parquet', 'wb') as f:
#     joblib.dump(results_all4, f)


import pandas as pd
import joblib


def load_data():
    try:
        with open('output/kmeans_0.csv', 'rb') as f:
            results = joblib.load(f)
            df = pd.DataFrame(results)
        return df
    except Exception as e:
        print("An error occurred while reading the Parquet file:", e)


def process_data_cluster(results):
    df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
    df_20_col = pd.read_parquet('output/seatunnal_20col.parquet')
    df_20_col_time = pd.concat([df_20_col, df_original['total_time']], axis=1)
    df = load_data()

    df2 = pd.DataFrame(df['2'].to_list(), columns=['num_2', 'score_2', 'lable_2'])
    df3 = pd.DataFrame(df['3'].to_list(), columns=['num_3', 'score_3', 'lable_3'])
    df4 = pd.DataFrame(df['4'].to_list(), columns=['num_4', 'score_4', 'lable_4'])

    new_df2 = pd.concat([df['df'], df2], axis=1)
    new_df3 = pd.concat([df['df'], df3], axis=1)
    new_df4 = pd.concat([df['df'], df4], axis=1)

    return new_df2, new_df3, new_df4

def process_dataframe(df_cluster, df_20_col, df_original):
    label_lists = []
    for i, row in df_cluster.iterrows():
        print("I", i)
        print("Row 0", row[0])
        print("Row 1", row[1])
        print("Row 2", row[2])
        print("Row3 ", row[3])
        selected_cols = [col for col in row[0] if col is not None]
        data_list = pd.DataFrame({
            'num': row[1],
            'score': row[2],
            'label': row[3]
        })
        df_concat = pd.concat([df_20_col[selected_cols], data_list, df_original['total_time']], axis=1)
        label_lists.append(df_concat)
    return label_lists


if __name__ == '__main__':
    df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
    df_20_col = pd.read_parquet('output/seatunnal_20col.parquet')
    re = load_data()
    results = load_data()
    print(results)

    new_df2, new_df3, new_df4 = process_data_cluster(results)
    print("Cluster2 :", new_df2)
    print("Cluster3 :", new_df3)
    print("Cluster4 :", new_df4)

    results_all2 = process_dataframe(new_df2, df_20_col, df_original)
    results_all3 = process_dataframe(new_df3, df_20_col, df_original)
    results_all4 = process_dataframe(new_df4, df_20_col, df_original)

    with open('output/results_all2.parquet', 'wb') as f:
        joblib.dump(results_all2, f)

    with open('output/results_all3.parquet', 'wb') as f:
        joblib.dump(results_all3, f)

    with open('output/results_all4.parquet', 'wb') as f:
        joblib.dump(results_all4, f)