import json
import time
from functools import partial
import os
import re
import joblib
import pandas as pd
import logging
import gc
import optuna
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from sklearn import model_selection


class OptunaRFOptimizer:
    def __init__(self, project_name="seatunnel", input_dir="output_resample", output_dir="output_resample/seatunnel-2", n_jobs_x=18):
        self.project_name = project_name
        self.INPUT_DIR = input_dir
        self.OUTPUT_DIR = output_dir
        self.n_jobs_x = n_jobs_x
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.CHECKPOINT_FILE = os.path.join(self.OUTPUT_DIR, f"{self.project_name}_processed_log.json")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(f"Running on project: {self.project_name}")

        # Ensure output_resample directory exists
        os.makedirs(self.INPUT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def objective(self, trial, x, y):
        n_estimators = trial.suggest_int('n_estimators', 100, 5000)
        max_depth = trial.suggest_int('max_depth', 1, 8)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 32)

        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=42)

        result = model_selection.cross_validate(rf, x, y, cv=5, n_jobs=self.n_jobs_x, scoring='f1')
        logging.info(result)
        scores = result['test_score']
        return np.mean(scores)

    def early_stopping_callback(self, study, trial, early_stopping_rounds):
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

    def load_processed_log(self):
        """Load processed feature groups from JSON file, handling empty or corrupted files."""
        if os.path.exists(self.CHECKPOINT_FILE):
            try:
                with open(self.CHECKPOINT_FILE, "r") as f:
                    data = f.read().strip()  # Read and strip any extra whitespace
                    if not data:  # Empty file case
                        return set()
                    return set(json.loads(data))  # Convert JSON list to a set
            except json.JSONDecodeError:
                logging.warning(f"Corrupted JSON detected in {self.CHECKPOINT_FILE}. Resetting file.")
                return set()  # Return empty set if corrupted
        return set()

    def save_processed_log(self, feature_group):
        """Save processed feature groups to JSON file."""
        processed = self.load_processed_log()
        processed.add(feature_group)  # Add the new feature group

        # Save back as a list
        with open(self.CHECKPOINT_FILE, "w") as f:
            json.dump(list(processed), f, indent=6)

        logging.info(f"Checkpoint updated: {feature_group}")

    def save_results(self, results):
        """Save results after processing a dataset."""
        if not results:
            return  # Avoid writing empty data

        df = pd.DataFrame(results)

        # Ensure valid filename
        feature_group_str = "".join(map(str, df['feature_group'][0]))
        feature_group_str = re.sub(r'[<>:"/\\|?*]', '_', feature_group_str)
        file_name = os.path.join(self.OUTPUT_DIR,
                                 f"{self.project_name}_{feature_group_str}_optuna_results_{self.timestamp}.pkl")

        df.to_pickle(file_name)
        logging.info(f"Results saved to {file_name}")

        # Update checkpoint log
        self.save_processed_log(feature_group_str)

    def find_best_parameter(self, dataset: pd.DataFrame):
        data = []
        time_start = time.time()

        # Load checkpoint JSON log
        processed_groups = self.load_processed_log()

        # Convert feature group to string for check
        feature_group_str = "_".join(dataset.columns.tolist())

        if feature_group_str in processed_groups:
            logging.info(f"Skipping already processed feature group: {feature_group_str}")
            return None

        x = dataset.drop(columns=["time_category"])
        x_fit = x.to_numpy()
        y_fit = dataset['time_category'].to_numpy()

        # Run Optuna optimization
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(
            lambda trial: self.objective(trial, x_fit, y_fit),
            n_trials=200,
            timeout=60,
            callbacks=[partial(self.early_stopping_callback, early_stopping_rounds=50)]
        )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial = study.best_trial
        result = trial.value
        result_state = trial.state
        best_params = trial.params

        time_end = time.time()
        time_min = (time_end - time_start) / 60

        data.append({
            "feature_group": feature_group_str,
            "best_params": best_params,
            "result": result,
            "total_trials": len(completed_trials),
            "result_state": result_state,
            'best_trial': trial.number,
            'time_min': time_min
        })

        # Save results after each dataset
        self.save_results(data)

        logging.info(f"Processed data: {data}")

        del x, x_fit, y_fit, study, completed_trials, trial, result, result_state, best_params
        gc.collect()

        return data

    def process_chunk(self, datasets_chunk):
        results = []
        for dataset in datasets_chunk:
            if isinstance(dataset, np.ndarray) and dataset.ndim == 3:
                dataset = pd.DataFrame(dataset.reshape(-1, dataset.shape[-1]))  # Flatten to 2D
            elif isinstance(dataset, list):
                dataset = pd.DataFrame(dataset)

            # Find the best parameters
            result = self.find_best_parameter(dataset)
            if result:  # Only add non-None results
                results.append(result)
            gc.collect()

        logging.info(f"Processed chunk of {len(datasets_chunk)} datasets and got {len(results)} results.")
        return results

    def parallel_optuna(self, datasets: list):
        logging.info(f"Starting Optuna for {len(datasets)} datasets...")

        # Parallel execution using Pool
        with Pool(processes=self.n_jobs_x) as pool:
            results = pool.map(self.process_chunk, datasets)

        # Check results
        logging.info(f"Total number of results: {len(results)}")

        return results

    def run(self, input_file):
        """Main execution function."""
        time_start = time.time()

        datasets = joblib.load(input_file)
        # datasets_test_data_each_5 = [dataset[:10] for dataset in datasets]

        find_best_para = self.parallel_optuna(datasets)

        time_end = time.time()
        time_sec = (time_end - time_start)
        time_min = (time_end - time_start) / 60
        time_day = (time_min / 86400)

        logging.info(f"Time taken: {time_sec:.4f} seconds")
        logging.info(f"Time taken: {time_min:.4f} minutes")
        logging.info(f"Time taken: {time_day:.4f} days")



# Main execution
if __name__ == '__main__':
    optimizer = OptunaRFOptimizer()
    optimizer.run(f"{optimizer.INPUT_DIR}/seatunnel_resamples_list_2_chunks18.pkl")
