import json
import time
from functools import partial
import os
import joblib
import numpy as np
import optuna
import pandas as pd
import logging
import gc
import lightgbm as lgb
from datetime import datetime
from multiprocessing import Pool
from sklearn import model_selection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "pulsar"

INPUT_DIR = os.path.join("/home/bill/origin-source-code-bill/dynamic/output_resample/resample_data")
OUTPUT_DIR = os.path.join("/home/bill/origin-source-code-bill/dynamic/output_resample/pulsar")
os.makedirs(INPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f"{project_name}_processed_log.json")

logging.info(f"Running on project: {project_name}")


# set up objective for using optuna
def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 100, 5000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 1, 8)
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

    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=18, scoring='f1')
    print(result)
    scores = result['test_score']
    score = np.mean(scores)
    return score


def early_stopping_callback(study, trial, early_stopping_rounds):
    current_trial = trial.number

    if current_trial < early_stopping_rounds:
        return

    best_trial_number = study.best_trial.number
    trials_without_improvement = current_trial - best_trial_number

    if trials_without_improvement >= early_stopping_rounds:
        logging.info(f"Early stopping triggered after {early_stopping_rounds} trials without improvement")
        logging.info(f"Best F1: {study.best_value}, found at trial {best_trial_number}")
        return {"state": optuna.trial.TrialState.PRUNED, "value": study.best_value, "params": study.best_params}
    else:
        return {"state": optuna.trial.TrialState.COMPLETE, "value": study.best_value, "params": study.best_params}


def load_processed_log():
    """Load processed feature groups from JSON file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))  # Convert JSON list to a set for quick lookup
    return set()


def find_best_parameter(dataset: list):
    data = []
    #  โหลด checkpoint JSON log
    processed_groups = load_processed_log()

    # แปลง feature group เป็น string เพื่อตรวจสอบ
    feature_group_str = "_".join(dataset.columns.tolist())

    if feature_group_str in processed_groups:
        logging.info(f"Skipping already processed feature group: {feature_group_str}")
        # ข้ามข้อมูลที่เคยรันแล้ว
        return None

    x = dataset.drop(columns=["time_category"])
    x_fit = x.to_numpy()
    y_fit = dataset['time_category'].to_numpy()

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(
        lambda trial: objective(trial, x_fit, y_fit),
        n_trials=200,
        timeout=60,
        callbacks=[partial(early_stopping_callback, early_stopping_rounds=50)]
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    trial = study.best_trial
    result = trial.value
    resultState = trial.state
    best_params = trial.params

    time_end = time.time()
    time_min = (time_end - time_start) / 60

    data.append({
        "feature_group": feature_group_str,
        "best_params": best_params,
        "result": result,
        "total_trials": len(completed_trials),
        "result_state": resultState,
        'best_trial': trial.number,
        'time_min': time_min
    })

    # Save results after each dataset
    save_results(data)

    # Clear memory
    del x, x_fit, y_fit, study, completed_trials, trial, result, resultState,best_params
    gc.collect()
    return data


def save_processed_log(feature_group):
    """Save processed feature groups to JSON file."""
    processed = load_processed_log()
    processed.add(feature_group)  # Add the new feature group

    # Convert set back to list for JSON storage
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(processed), f, indent=6)

    del processed
    gc.collect()


def save_results(results):
    # Save results after processing a dataset and update the checkpoint.
    if not results:
        return  # Avoid writing empty data

    df = pd.DataFrame(results)

    # Ensure valid filename
    feature_group_str = "".join(map(str, df['feature_group'][0]))
    file_name = os.path.join(OUTPUT_DIR, f"{project_name}_{feature_group_str}_optuna_results_{timestamp}.pkl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output_resample directory exists
    df.to_pickle(file_name)

    # Update checkpoint log
    save_processed_log(feature_group_str)

    logging.info(f"Results saved to {file_name}")
    logging.info(f"Checkpoint updated: {feature_group_str}")

    del df, feature_group_str
    gc.collect()


def process_chunk(datasets_chunk):
    results = []
    for dataset in datasets_chunk:
        # เช็คว่า dataset เป็น array 3D แล้วแปลงเป็น DataFrame
        if isinstance(dataset, np.ndarray) and dataset.ndim == 3:
            # แปลงจาก array 3D เป็น DataFrame โดยเลือกมิติที่ต้องการ เช่น เลือกมิติสุดท้าย (1348, 8)
            dataset = pd.DataFrame(dataset.reshape(-1, dataset.shape[-1]))  # แปลงเป็น 2D
        elif isinstance(dataset, list):
            dataset = pd.DataFrame(dataset)

        # เรียกใช้งานฟังก์ชัน find_best_parameter
        result = find_best_parameter(dataset)
        results.append(result)

        gc.collect()

    return results


def parallel_optuna(datasets: list):
    logging.info(f"Starting Optuna for {len(datasets)} datasets...")

    # Using Pool for parallel execution
    with Pool(processes=18) as pool:
        results = pool.map(process_chunk, datasets)
    return results


# main execution
if __name__ == '__main__':
    # Load the data
    time_start = time.time()
    # datasets = joblib.load(f'{INPUT_DIR}/pulsar_resampled_data_20250318_015313.pkl')
    datasets = joblib.load((f'{INPUT_DIR}/pulsar_resamples_list_2_chunks6.pkl'))
    # print(datasets)
    find_best_para = parallel_optuna(datasets)

    time_end = time.time()
    time_sec = (time_end - time_start)
    time_min = (time_end - time_start) / 60
    time_day = (time_min / 86400)

    logging.info(f"Time taken: {time_sec:.4f} seconds")
    logging.info(f"Time taken: {time_min:.4f} minutes")
    logging.info(f"Time taken: {time_day:.4f} days")
