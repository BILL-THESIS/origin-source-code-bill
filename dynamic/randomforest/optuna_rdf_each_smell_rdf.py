import time
import os
import joblib
import numpy as np
import optuna
import logging
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn import model_selection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "pulsar"

INPUT_DIR = os.path.join("../lightgbm/output")
OUTPUT_DIR = os.path.join("../lightgbm/output")
os.makedirs(INPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f"{project_name}_processed_log.json")

logging.info(f"Running on project: {project_name}")


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

    result = model_selection.cross_validate(rf, x, y, cv=5, n_jobs=4, scoring='f1')
    logging.info(result)
    scores = result['test_score']
    return np.mean(scores)


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


# start optuna
def find_best_parameter(datasets: list, dataset_name: str):
    data = []
    # Set up time and line notification
    for i, dataset in enumerate(datasets):

        x_fit = dataset[0].to_numpy()
        y_fit = dataset[1].to_numpy()

        # Find best parameter
        # try:
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, x_fit, y_fit),
            n_trials=200,
            timeout=60,
        )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        trial = study.best_trial
        result = trial.value
        resultState = trial.state
        best_params = trial.params

        data.append({
            "feature_group": dataset[0],
            "best_params": best_params,
            "result": result,
            "total_trials": len(completed_trials),
            "result_state": resultState,
            'best_trial': trial.number
        })

    # Save the model and results path
    file_name = f"{OUTPUT_DIR}/{dataset_name}_each_smells_rdf.pkl"
    joblib.dump(data, file_name)

    return data


# main execution
if __name__ == '__main__':
    # Load the data
    time_start = time.time()

    # datasets = joblib.load((f'{INPUT_DIR}/seatunnel_resamples_each_smell.pkl'))
    datasets = joblib.load((f'{INPUT_DIR}/ozone_resamples_each_smell.pkl'))
    # datasets = joblib.load((f'{INPUT_DIR}/pulsar_resamples_each_smell.pkl'))

    # Find best parameter
    data = find_best_parameter(datasets, "ozone")

    time_end = time.time()
    time_sec = (time_end - time_start)
    time_min = (time_end - time_start) / 60
    time_day = (time_min / 86400)

    logging.info(f"Time taken: {time_sec:.4f} seconds")
    logging.info(f"Time taken: {time_min:.4f} minutes")
    logging.info(f"Time taken: {time_day:.4f} days")
