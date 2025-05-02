import time
from functools import partial
import os
import joblib
import numpy as np
import optuna
import pandas as pd
import logging
import gc
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from multiprocessing import Pool
from sklearn import model_selection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# set up objective for using optuna
def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 100, 5000)
    max_depth = trial.suggest_int('max_depth', 1, 8)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 32)

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    result = model_selection.cross_validate(rf, x, y, cv=5, n_jobs=18, scoring='f1')
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


def chunk_list(lst, n_chunks):
    # Split a list into n roughly equal-sized chunks
    chunk_size = int(np.ceil(len(lst) / n_chunks))
    return [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]


def parallel_optuna(datasets: list):
    logging.info(f"Starting Optuna for {len(datasets)} datasets...")

    # Using Pool for parallel execution
    with Pool(processes=18) as pool:
        results = pool.map(find_best_parameter, datasets)
    return results


# main execution
if __name__ == '__main__':
    project_name = "pulsar"

    INPUT_DIR = os.path.join("../02.resample_data/output_resample")
    OUTPUT_DIR = os.path.join("output_randomforest/")
    os.makedirs(INPUT_DIR, exist_ok=True)

    logging.info(f"Running on project: {project_name}")

    # Load the data
    time_start = time.time()
    datasets = joblib.load((f'{INPUT_DIR}/pulsar_resampled_data.pkl'))

    datasets = chunk_list(datasets, 18)

    find = parallel_optuna(datasets)
    list_l = []
    for data_list in find:
        for dataset in data_list:
            list_l.append(dataset)
    df = pd.DataFrame(list_l)

    joblib.dump(df, f'{OUTPUT_DIR}{project_name}_optuna_result_rdf.pkl')

    time_end = time.time()
    time_sec = (time_end - time_start)
    time_min = (time_end - time_start) / 60
    time_day = (time_min / 86400)

    logging.info(f"Time taken: {time_sec:.4f} seconds")
    logging.info(f"Time taken: {time_min:.4f} minutes")
    logging.info(f"Time taken: {time_day:.4f} days")
