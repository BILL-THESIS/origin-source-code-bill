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

    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=4, scoring='f1')
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


def find_best_parameter(datasets: list):
    data = []
    time_start = time.time()

    for i, dataset in enumerate(datasets):
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
            "feature_group": x.columns.tolist(),
            "best_params": best_params,
            "result": result,
            "total_trials": len(completed_trials),
            "result_state": resultState,
            'best_trial': trial.number,
            'time_min': time_min
        })

    # Clear memory
    del x, x_fit, y_fit, study, completed_trials, trial, result, resultState, best_params
    gc.collect()
    return data


def parallel_optuna(datasets: list):
    logging.info(f"Starting Optuna for {len(datasets)} datasets...")

    # Using Pool for parallel execution
    with Pool(processes=18) as pool:
        results = pool.map(find_best_parameter, [[dataset] for dataset in datasets])
    return results


# main execution
if __name__ == '__main__':
    # Load the data
    time_start = time.time()
    project_name = "seatunnel"

    INPUT_DIR = os.path.join("../02.resample_data/output_resample")
    OUTPUT_DIR = os.path.join("output_lightgbm/")
    os.makedirs(INPUT_DIR, exist_ok=True)

    logging.info(f"Running on project: {project_name}")

    datasets = joblib.load((f'{INPUT_DIR}/seatunnal_resampled_group1.pkl'))

    find = parallel_optuna(datasets)
    list_l = []
    for data_list in find:
        for dataset in data_list:
            list_l.append(dataset)
    df = pd.DataFrame(list_l)

    joblib.dump(df, f'{OUTPUT_DIR}seatunnel_optuna_result_group1.pkl')

    time_end = time.time()
    time_sec = (time_end - time_start)
    time_min = (time_end - time_start) / 60
    time_day = (time_min / 86400)

    logging.info(f"Time taken: {time_sec:.4f} seconds")
    logging.info(f"Time taken: {time_min:.4f} minutes")
    logging.info(f"Time taken: {time_day:.4f} days")
