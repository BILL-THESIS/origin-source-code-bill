def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import joblib
import numpy as np
import optuna
import pandas as pd
import logging
import gc
import lightgbm as lgb
from datetime import datetime
from joblib import Parallel, delayed
from sklearn import model_selection


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

INPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output_resample/output_resample")
OUTPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output_resample/resample_data")
os.makedirs(INPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

n_jobs = 2

# set up objective for using optuna
def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 100, 5000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 1, 8)
    # ดูจากแถวของข้อมูลที่มีอยู่
    min_child_samples = trial.suggest_int('min_child_samples', 2, 32)
    min_child_weight = trial.suggest_int('min_child_weight', 2, 32)
    subsample = trial.suggest_float('subsample', 0.1, 1.0)

    gbm = lgb.LGBMClassifier(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             max_depth=max_depth,
                             min_child_samples=min_child_samples,
                             min_child_weight=min_child_weight,
                             subsample=subsample,
                             random_state=42,
                             num_threads=1,
                             verbosity=-1)

    # กลับไปดูบน Doc มีอยู่แล้ว
    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=n_jobs, scoring='f1')
    print(result)
    scores = result['test_score']
    score = np.mean(scores)
    return score


# start optuna
def find_best_parameter(datasets: list):
    data = []
    for dataset in datasets:
        x = dataset.drop(columns=["time_category"])
        x_fit = x.to_numpy()
        y_fit = dataset['time_category'].to_numpy()

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, x_fit, y_fit),
            n_trials=20
            # n_trials=1000,
            # timeout=600
        )

        trial = study.best_trial
        result = trial.value
        best_params = trial.params

        data.append({
            'feature_group': x.columns.tolist(),
            'best_params': best_params,
            'result': result
        })

    return data


def parallel_optuna(datasets: list):
    logging.info(f"Starting Optuna for {len(datasets)} datasets...")
    datasets_chunk = datasets
    # Process each chunk in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group_chunk)(chunk)
        for chunk in datasets_chunk
    )
    results = [res for res in results if res is not None]

    # save the results to a file
    output_filepath = os.path.join(OUTPUT_DIR, f"{project_name}_optuna_results_{timestamp}.pkl")
    joblib.dump(results, output_filepath)
    logging.info(f"Optuna results saved to {output_filepath}")
    return results


def process_group_chunk(datasets_chunk):
    logging.info(f"Processing feature group chunk")
    chunk_results = []
    for idx, dataset in enumerate(datasets_chunk):
        best_params = find_best_parameter([dataset])
        if best_params:
            # append the best parameters to the chunk results
            chunk_results.extend(best_params)
    return chunk_results


# main execution
if __name__ == '__main__':
    # Load the data
    datasets = joblib.load(
        '/Users/bill/origin-source-code-bill/dynamic/output_resample/resample_data/seatunnel_resampled_data_20250227_104730.pkl')

    find_best_para = parallel_optuna(datasets)
