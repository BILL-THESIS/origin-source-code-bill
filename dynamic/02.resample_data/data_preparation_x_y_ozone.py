import gc
import logging
import os
import pickle
from datetime import datetime
import random

from joblib import Parallel, delayed
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "ozone"

INPUT_DF = os.path.join("../output/")
INPUT_CORRE = os.path.join("../01.variable_clustering/output_variable")
OUTPUT_DF = os.path.join("output_resample/")
os.makedirs(INPUT_DF, exist_ok=True)
os.makedirs(INPUT_CORRE, exist_ok=True)

logging.info(f"Running on project: {project_name}")

# File paths
INPUT_FILEPATH = os.path.join(INPUT_DF, f"{project_name}_cut_time.pkl")
GROUP_FILEPATH = os.path.join(INPUT_CORRE, f"{project_name}_combinations.pkl")


def load_data(input_filepath=INPUT_FILEPATH, group_filepath=GROUP_FILEPATH):
    logging.info("Loading data...")

    # data features groups
    with open(group_filepath, 'rb') as f:
        feature_groups = pickle.load(f)

    # Load dataset
    data = pd.read_pickle(input_filepath)

    # Reduce memory usage
    for col in data.select_dtypes(include=['int64']):
        data[col] = pd.to_numeric(data[col], downcast='integer')
    for col in data.select_dtypes(include=['float64']):
        data[col] = pd.to_numeric(data[col], downcast='float')

    logging.info(f"Loaded dataset with {len(data)} rows and {len(feature_groups)} feature groups.")
    return data, feature_groups


def preprocess_time_category(data):
    #Add time category column based on quantiles of total_hours
    logging.info("Processing time category...")

    data['total_time'] = pd.to_timedelta(data['total_time']).fillna(pd.Timedelta(0))
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600

    q3 = data['total_hours'].quantile(0.75)

    if q3 >= 0:
        data['time_category'] = pd.cut(
            data['total_hours'], bins=[-float('inf'), q3, float('inf')], labels=[0, 1], right=False
        )
    else:
        logging.warning("q3 is 0, setting all time_category to 0")
        data['time_category'] = 0

    data.drop(columns=['total_time', 'total_hours'], inplace=True)
    gc.collect()
    logging.info("Time category processed.")
    return data

def resample_feature_group(feature_group, data_perpa_x):
    df_list = []

    X = data_perpa_x[data_perpa_x.columns.intersection(feature_group)].fillna(0)
    y = data_perpa_x['time_category']

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

    # Save the resampled data to a file
    df_list.append(resampled_data)

    del X, y, X_resampled, y_resampled
    gc.collect()
    return df_list


if __name__ == "__main__":
    data_perpa_x, feature_groups = load_data()
    data_perpa_x = preprocess_time_category(data_perpa_x)

    list_df = []
    for i, feature_group in enumerate(feature_groups):
        X = data_perpa_x[data_perpa_x.columns.intersection(feature_group)].fillna(0)
        y = data_perpa_x['time_category']

        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
        list_df.append(resampled_data)

        # Save the resampled data
        output_file = os.path.join(OUTPUT_DF, f"{project_name}_resampled_data.pkl")
        joblib.dump(list_df, output_file)
