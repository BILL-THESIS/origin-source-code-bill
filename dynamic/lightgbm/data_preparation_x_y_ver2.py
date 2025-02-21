import gc
import logging
import os
import pickle
from datetime import datetime
from joblib import Parallel, delayed
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

OUTPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

# File paths
INPUT_FILEPATH = os.path.join(OUTPUT_DIR, f"{project_name}_compare.pkl")
GROUP_FILEPATH = os.path.join(OUTPUT_DIR, f"{project_name}_correlation_group_13360.pkl")


def load_data(input_filepath=INPUT_FILEPATH, group_filepath=GROUP_FILEPATH):
    """Load dataset and feature groups."""
    logging.info("Loading data...")

    # Load feature groups
    with open(group_filepath, 'rb') as f:
        feature_groups = pickle.load(f)

    # Load dataset
    data = pd.read_pickle(input_filepath)

    # ลดขนาดของตัวแปรประเภทตัวเลข
    for col in data.select_dtypes(include=['int64']):
        data[col] = pd.to_numeric(data[col], downcast='integer')
    for col in data.select_dtypes(include=['float64']):
        data[col] = pd.to_numeric(data[col], downcast='float')

    logging.info("Data and feature groups loaded successfully.")
    return data, feature_groups


def preprocess_time_category(data):
    """Add time category column based on quantiles of total_hours."""
    logging.info("Processing time category...")

    # แปลง total_time เป็น timedelta และกำจัดค่า NaT
    data['total_time'] = pd.to_timedelta(data['total_time']).fillna(pd.Timedelta(0))

    # แปลงเวลาเป็นชั่วโมง
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600

    # คำนวณ quantile ที่ 75%
    q3 = data['total_hours'].quantile(0.75)

    # ตรวจสอบว่าค่า q3 มีค่ามากกว่า 0 หรือไม่
    if q3 > 0:
        data['time_category'] = pd.cut(
            data['total_hours'], bins=[-float('inf'), q3, float('inf')], labels=[0, 1], right=False
        )
    else:
        logging.warning("q3 is 0, setting all time_category to 0")
        data['time_category'] = 0

    # ลบคอลัมน์ที่ไม่จำเป็น
    data.drop(columns=['total_time', 'total_hours'], inplace=True)
    gc.collect()
    logging.info("Time category processed.")
    return data


def resample_data(data_perpa_x, feature_groups):
    logging.info("Resampling data...")

    # select features X in list of feature groups and target y in data_perpa_x columns 'time_category'
    list_resampled_data = []

    for feature_group in feature_groups:
        logging.info(f"Count time of resample : {feature_group}")
        logging.info(f"Feature group index is {feature_groups.index(feature_group)}")

        X = data_perpa_x[data_perpa_x.columns.intersection(feature_group)]
        X = X.fillna(0)
        y = data_perpa_x['time_category']

        # data SMOTE resampling x and y
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info(f"Resampled data shape: {X_resampled.shape}, {y_resampled.shape}")

        # save resampled data
        resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

        # Convert Data Frame to Dictionary
        resampled_data_dict = resampled_data.to_dict()

        # Save resampled data to pickle
        list_resampled_data.append(resampled_data_dict)

        # save resampled data to pickle
        f = f"{OUTPUT_DIR}/{project_name}_resampled_data_{timestamp}.pkl"
        joblib.dump(list_resampled_data, f)
        logging.info(f"Resampled data saved to: {f}")

        del X, y, X_resampled, y_resampled, resampled_data, resampled_data_dict
        gc.collect()

    return list_resampled_data


if __name__ == "__main__":
    data_perpa_x, feature_groups = load_data()
    feature_groups = feature_groups[:100]
    data_perpa_x = preprocess_time_category(data_perpa_x)
    resampled_data = resample_data(data_perpa_x, feature_groups)
    logging.info("Start parallel processing with 3 CPU cores for resampling data.")
    logging.info("Parallel processing completed.")
    logging.info("Data preparation completed.")
