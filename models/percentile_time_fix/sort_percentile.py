import os
import joblib
import pandas as pd


def sort_percentile_f1(df, project_name):
    # Create a boolean mask for rows where the specified column's value is > 0.6 and < 0.9
    mask = (df['f1_smote'] > 0.6) & (df['f1_smote'] < 0.9)

    # Invert the mask to keep rows that do not meet the condition
    df_filtered_less_point = df[~mask]
    df_filtered_more_point = df[mask]

    df_filtered_more_point.to_pickle(f'../output/{project_name}_split_f1_smote_time_class3.pkl')

    return df_filtered_less_point, df_filtered_more_point


if __name__ == '__main__':
    # Load the data
    with open(os.path.join('../output/pulsar_GBC_class_time3_smote_new.parquet'), 'rb') as f:
        pulsar_class_3_smote = joblib.load(f)

    with open(os.path.join('../output/ozone_GBC_class_time_3_smote_new.parquet'), 'rb') as f:
        ozone_class_3_smote = joblib.load(f)

    with open(os.path.join('../output/seatunnel_GBC_class_time3_smote.parquet'), 'rb') as f:
        seatunnel_class_3_smote = joblib.load(f)

    pulsar_unutilized, pulsar_utilize = sort_percentile_f1(pulsar_class_3_smote, 'pulsar')
    print('pulsar unutilized describe', pulsar_unutilized.describe())
    print("\n")
    print('pulsar utilize describe', pulsar_utilize.describe())

    ozone_unutilized, ozone_utilize = sort_percentile_f1(ozone_class_3_smote, 'ozone')

    seatunnel_unutilized, seatunnel_utilize = sort_percentile_f1(seatunnel_class_3_smote, 'seatunnel')