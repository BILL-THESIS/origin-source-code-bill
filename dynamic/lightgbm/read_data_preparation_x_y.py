import gc
import logging
import os
import pickle
from datetime import datetime
from joblib import Parallel, delayed
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np

from dynamic.lightgbm.lightgbm_optuna import INPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

INPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output_resample/resample_data")
os.makedirs(INPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

# File paths
INPUT_FILEPATH = os.path.join(INPUT_DIR, f"{INPUT_DIR}/seatunnel_resampled_data_20250223_111633.pkl")

def load_data(input_filepath=INPUT_FILEPATH):
    """Load dataset and feature groups."""
    logging.info("Loading data...")
    # Load dataset
    data = joblib.load(input_filepath)

    logging.info("Data and feature groups loaded successfully.")
    return data


if __name__ == "__main__":
    data = load_data()
