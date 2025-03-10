import itertools

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
            logging.info(f"Successfully loaded model from {file_path}")
            return model
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while loading the model from {file_path}: {e}")


def robust_outlier_detection(df):
    data = df['total_time']

    # median of the absolute deviations from the median (MAD)
    median = data.median()
    print("Median: ", median)

    mad = np.abs(data - median).median()
    print("MAD: ", mad)

    MADN = (mad / 0.6745)
    print("MADN: ", MADN)

    threshold = 2.24
    outlier = (data - median).abs() / MADN > threshold
    print("Sum outliers :", outlier.sum())

    # divided the dataset into two parts: normal and outliers
    df_outliers = df[outlier]
    df_normal = df[~outlier]

    return df_outliers, df_normal


def calculate_percentiles(date_series):
    percentiles = [np.percentile(date_series, range(0, 101))]
    percentile_df = pd.DataFrame(percentiles).T
    percentile_df.values.flatten()
    return percentile_df


def set_index_combinations_percentiles(percentile_list):
    list_each_percentile = []
    for points in percentile_list:
        df = pd.DataFrame(points)
        list_each_percentile.append(df)
    return list_each_percentile


def table_time_fix_percentile(list_each_percentile):
    time01_list = []
    time12_list = []
    index_time_01_list = []
    index_time_12_list = []

    for points in list_each_percentile:
        sort_point = points.sort_values(by=[0], ascending=True)

        time_01 = sort_point.iloc[0][0]
        # print("time_01", time_01)
        time_12 = sort_point.iloc[1][0]
        # print("time_12", time_12)

        index_time01 = sort_point.index[0]
        # print("index_time01", index_time01)
        index_time12 = sort_point.index[1]
        # print("index_time12", index_time12)

        time01_list.append(time_01)
        time12_list.append(time_12)

        index_time_01_list.append(index_time01)
        index_time_12_list.append(index_time12)

    df_time_point = pd.DataFrame(
        {'index_time01': index_time_01_list, 'index_time12': index_time_12_list, 'time01': time01_list,
         'time12': time12_list})
    return df_time_point


def divide_time_class_2(df_original, df_time_point):
    results = []
    for index, row in df_time_point.iterrows():
        # Create a copy of the DataFrame for the current percentile
        time01 = row['time01']
        time12 = row['time12']
        time_fix_hours = df_original['total_time']
        # time_fix_hours = df_original['total_time'].dt.total_seconds() / 3600

        values_time = []
        for time_i in time_fix_hours:
            if time_i <= time01:
                values_time.append(0)
                # print(f"time modify :: {time_i} < time01 :: {time01}")
            elif (time_i > time01) & (time_i < time12):
                values_time.append(1)
                # print(f"time01 :: {time01} >= time modify :: {time_i} < time12 :: {time12}")
            else:  # time_i >= time12
                values_time.append(2)
                # print(f"time modify :: {time_i} >=  time12 :: {time12}")

        # Create the 'time_class' column directly during iteration
        df_original['time_class'] = values_time
        df_original['index_time01'] = row['index_time01']
        df_original['time_01'] = row['time01']
        df_original['index_time12'] = row['index_time12']
        df_original['time_12'] = row['time12']

        # Append the modified DataFrame to results
        results.append(df_original.copy())
        # Avoid modifying the original

    return results


if __name__ == '__main__':
    model_original = pd.read_pickle('../../Github/output/kafka_filtered_final_api.pkl')

    # divide the dataset into normal and outliers
    df_outliers, df_normal = robust_outlier_detection(model_original)

    df_normal.to_parquet('../output/kafka_filtered_robust_outlier.parquet')

    # prepare the data time modify to calculate the percentiles
    percentiles_normal = calculate_percentiles(df_normal['total_time'])
    percentiles_outliers = calculate_percentiles(df_outliers['total_time'])

    # combinations of percentiles to divide time class for 3 classes
    time_point_list = list(itertools.combinations(percentiles_normal.iloc, 2))
    df_time_point_index = set_index_combinations_percentiles(time_point_list)
    df_time_point_sort = table_time_fix_percentile(df_time_point_index)

    # Plotting total_time_hours
    plt.figure(figsize=(10, 6))
    sns.histplot(df_normal['total_time_hours'], bins=50, kde=True)
    plt.xlabel('Total Time (hours)')
    plt.ylabel('Frequency')
    plt.title('Outliers Kafka Repository')
    plt.show()


    # save the model
    end = df_normal['merge_commit_sha'].drop_duplicates()
    end.to_csv('../output/tracking_api_to_sonar/kafka_filtered_robust_outlier_end.txt', header=True, index=False)
    start = df_normal['base.sha'].drop_duplicates()
    start.to_csv('../output/tracking_api_to_sonar/kafka_filtered_robust_outlier_start.txt', header=True, index=False)
