import os
from token import INDENT

import joblib
import numpy as np
import optuna
import pandas as pd
import logging
import gc
import lightgbm as lgb
from datetime import datetime
from sklearn import model_selection
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

INPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output/output")
OUTPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output/resample_data")
os.makedirs(INPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")


# set up objective for using optuna
def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 500, 5000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_child_samples = trial.suggest_int('min_child_samples', 128, 512)
    min_child_weight = trial.suggest_int('min_child_weight', 64, 256)
    subsample = trial.suggest_float('subsample', 0.1, 1.0)

    gbm = lgb.LGBMClassifier(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             max_depth=max_depth,
                             min_child_samples=min_child_samples,
                             min_child_weight=min_child_weight,
                             subsample=subsample,
                             random_state=42,
                             n_jobs=3)

    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=3, scoring='f1')
    print(result)
    scores = result['test_score']
    score = np.mean(scores)
    return score


# start optuna
def find_best_parameter(datasets: list):
    data = []  # Change to a list
    for dataset in datasets:
        x = dataset.drop(columns=["time_category"])
        x_fit = x.to_numpy()
        y_fit = dataset['time_category'].to_numpy()

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, x_fit, y_fit),
            n_trials=5,
        )
        trial = study.best_trial
        result = trial.value
        best_params = trial.params

        data.append({
            'feature_group': x.columns.tolist(),
            'best_params': best_params,
            'result': result
        })  # Append results instead of overwriting

    return data



def parallel_optuna(datasets: list):
    logging.info(f"Starting Optuna for {len(datasets)} datasets...")
    datasets_chunk = datasets
    results = Parallel(n_jobs=4)(
        delayed(process_group_chunk)(chunk)  # Pass chunk directly
        for chunk in datasets_chunk
    )
    results = [res for res in results if res is not None]

    #save the results to a file
    output_filepath = os.path.join(OUTPUT_DIR, f"{project_name}_optuna_results_{timestamp}.pkl")
    joblib.dump(results, output_filepath)
    logging.info(f"Optuna results saved to {output_filepath}")
    return results


def process_group_chunk(datasets_chunk):
    logging.info(f"Processing feature group chunk")

    chunk_results = []
    for idx, dataset in enumerate(datasets_chunk):
        best_params = find_best_parameter([dataset])  # Pass as list
        if best_params:
            chunk_results.extend(best_params)  # Use extend instead of append to avoid nested lists
    return chunk_results



# main execution
if __name__ == '__main__':
    # Load the data
    datasets = joblib.load('/Users/bill/origin-source-code-bill/dynamic/output/resample_data/seatunnel_resampled_data_20250224_113201.pkl')

    find_best_para = parallel_optuna(datasets)
