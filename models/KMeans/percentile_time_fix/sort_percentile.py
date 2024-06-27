import os
import joblib
import pandas as pd


def sort_percentile_f1(df):
    # Create a boolean mask for rows where the specified column's value is > 0.6 and < 0.9
    mask = (df['f1_smote'] > 0.6) & (df['f1_smote'] < 0.9)

    # Invert the mask to keep rows that do not meet the condition
    df_filtered_less_point = df[~mask]
    df_filtered_more_point = df[mask]

    return df_filtered_less_point, df_filtered_more_point


if __name__ == '__main__':
    # Load the data
    with open(os.path.join('../../KMeans/output/class_time_2_smote.parquet'), 'rb') as f:
        df_class_2_smote = joblib.load(f)

    with open(os.path.join('../../KMeans/output/class_time_3_smote.parquet'), 'rb') as f:
        df_class_3_smote = joblib.load(f)

    print('df_class_3_smote describe', df_class_3_smote.describe())
    print("\n")

    # Convert 'time01' and 'time12' to total seconds in hours
    df_class_3_smote['time01'] = pd.to_timedelta(df_class_3_smote['time01']).dt.total_seconds() / 3600
    df_class_3_smote['time12'] = pd.to_timedelta(df_class_3_smote['time12']).dt.total_seconds() / 3600

    df_unutilized, df_utilize = sort_percentile_f1(df_class_3_smote)
    print('df_less describe', df_unutilized.describe())
    print("\n")
    print('df_more describe', df_utilize.describe())

    sort_df_more_time01 = df_utilize.sort_values(by='time01', ascending=False)
    sort_df_more_time12 = df_utilize.sort_values(by='time12', ascending=False)

    df_utilize.to_pickle('../../KMeans/output/class_time_3_smote_utilize.pkl')