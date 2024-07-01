import numpy as np
import pandas as pd
import joblib


def robust_outlier_detection(df):
    data = df['f1_macro']

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


if __name__ == '__main__':
    with open('../output/class_time_3_normal.parquet', 'rb') as file:
        model = joblib.load(file)

    df_outliers, df_normal = robust_outlier_detection(model)

