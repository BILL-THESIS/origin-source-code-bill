import time
import os
import joblib
import numpy as np
import optuna
import logging
import lightgbm as lgb
from datetime import datetime
from sklearn import model_selection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "pulsar"

INPUT_DIR = os.path.join("../02.resample_data/output_resample")
OUTPUT_DIR = os.path.join("../02.resample_data/output_resample")
os.makedirs(INPUT_DIR, exist_ok=True)


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

    result = model_selection.cross_validate(gbm, x, y, cv=5,  scoring='f1')
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
    file_name = f"{OUTPUT_DIR}/{dataset_name}_each_smells.pkl"
    joblib.dump(data, file_name)

    return data


# main execution
if __name__ == '__main__':
    # Load the data
    time_start = time.time()

    datasets = joblib.load((f'{INPUT_DIR}/pulsar_resampled_data_each_smell.pkl'))

    # Find best parameter
    data = find_best_parameter(datasets, "seatunnel")

    time_end = time.time()
    time_sec = (time_end - time_start)
    time_min = (time_end - time_start) / 60
    time_day = (time_min / 86400)

    logging.info(f"Time taken: {time_sec:.4f} seconds")
    logging.info(f"Time taken: {time_min:.4f} minutes")
    logging.info(f"Time taken: {time_day:.4f} days")
