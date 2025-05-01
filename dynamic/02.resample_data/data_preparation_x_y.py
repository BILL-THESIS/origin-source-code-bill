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
project_name = "pulsar"

INPUT_DIR = os.path.join("/dynamic/output/")
INPUT_DIR2 = os.path.join("/dynamic/output/output/")
OUTPUT_DIR = os.path.join("/dynamic/output/output")
os.makedirs(INPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

# File paths
INPUT_FILEPATH = os.path.join(INPUT_DIR, f"{project_name}_compare.pkl")
# GROUP_FILEPATH = os.path.join(INPUT_DIR, f"{project_name}_correlation_group_13360.pkl")
GROUP_FILEPATH = os.path.join(INPUT_DIR2, f"pulsar_correlation_group.pkl")


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

    if q3 > 0:
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


def chunk_list(lst, n_chunks):
    #Split a list into n roughly equal-sized chunks
    chunk_size = int(np.ceil(len(lst) / n_chunks))
    return [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]


def resample_feature_group(feature_group, data_perpa_x, feature_group_idx):
    #Resample data for a given feature group using SMOTE
    logging.info(f"Processing feature group {feature_group_idx + 1}")

    X = data_perpa_x[data_perpa_x.columns.intersection(feature_group)].fillna(0)
    y = data_perpa_x['time_category']

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

    del X, y, X_resampled, y_resampled
    gc.collect()

    return resampled_data


def parallel_resampling(data_perpa_x, feature_groups, num_cores=2):

    #Parallelize SMOTE resampling across feature groups
    logging.info(f"Splitting {len(feature_groups)} feature groups into {num_cores} chunks for parallel processing...")
    # Split feature groups into chunks
    feature_group_chunks = chunk_list(feature_groups, num_cores)
    # Process each chunk in parallel
    results = Parallel(n_jobs=num_cores)(
        delayed(process_feature_group_chunk)(chunk, data_perpa_x, chunk_id)
        for chunk_id, chunk in enumerate(feature_group_chunks)
    )

    # Merge all resampled data
    results = [res for res in results if res is not None]

    # Save the resampled data
    output_file = os.path.join(OUTPUT_DIR, f"{project_name}_resampled_data_{timestamp}.pkl")
    joblib.dump(results, output_file)
    logging.info(f"Resampled data saved to: {output_file}")

    return results


def process_feature_group_chunk(feature_group_chunk, data_perpa_x, chunk_id):
    logging.info(f"Processing feature group chunk {chunk_id + 1}")
    chunk_results = []
    for idx, feature_group in enumerate(feature_group_chunk):
        resampled_data = resample_feature_group(feature_group, data_perpa_x, idx)
        # resampled_data = dict(features=resampled_data.drop(columns=["time_category"]), target=resampled_data["time_category"])
        if resampled_data is not None:
            chunk_results.append(resampled_data)

    return chunk_results

if __name__ == "__main__":
    data_perpa_x, feature_groups = load_data()
    # test with a random sample of 10 feature groups
    random_sample_feature_groups = random.sample(feature_groups, 10 )
    data_perpa_x = preprocess_time_category(data_perpa_x)

    logging.info("Starting parallel processing over feature groups...")
    # resampled_data = parallel_resampling(data_perpa_x, random_sample_feature_groups, num_cores=2)
    logging.info("Parallel processing completed.")
