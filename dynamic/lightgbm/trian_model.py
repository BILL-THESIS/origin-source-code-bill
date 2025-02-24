import os
import joblib
import logging
import gc
import lightgbm as lgb
from datetime import datetime

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

INPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output/resample_data")
OUTPUT_DIR = os.path.join("/Users/bill/origin-source-code-bill/dynamic/output/resample_data")
os.makedirs(INPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

File_best_param = f"{INPUT_DIR}/seatunnel_optuna_results_20250224_113221.pkl"
File_resample_data = f"{INPUT_DIR}/seatunnel_resampled_data_20250224_113201.pkl"

n_jobs = 3


def extract_features_targets(data_resample):
    X_list, y_list = [], []
    for chunk in data_resample:
        for c_value in chunk:
            # Assume 'features' key holds X data
            X_list.append(c_value.drop(columns=["time_category"]))
            # Assume 'target' key holds y labels
            y_list.append(c_value["time_category"])
    return X_list, y_list


def extract_best_params(data_best_param):
    best_params = []
    for chunk in data_best_param:
        for d_value in chunk:
            best_params.append((d_value['feature_group'], d_value['best_params']))
    return best_params


def train_model(X_list, y_list, feature_best_param):
    result = []
    for idx, feature_best in enumerate(feature_best_param):
        print("feature_best: ", feature_best)
        feature_group_X = feature_best[0]
        print("feature_best['feature_group']: ", feature_best[0])
        feature_group_param = feature_best[1]
        print("feature_best['best_params']: ", feature_best[1])

        for idx, X in enumerate(X_list):
            if all(feature in X.columns for feature in feature_group_X):
                print(f"Values of X_list[{idx + 1}]:\n", X.columns)
                print(f"Feature: {feature_group_X}")

                x_fit = X.to_numpy()
                y_fit = y_list[idx].to_numpy()

                # split data
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # model
                model = lgb.LGBMClassifier(
                    n_estimators=feature_group_param['n_estimators'],
                    learning_rate=feature_group_param['learning_rate'],
                    max_depth=feature_group_param['max_depth'],
                    min_child_samples=feature_group_param['min_child_samples'],
                    min_child_weight=feature_group_param['min_child_weight'],
                    subsample=feature_group_param['subsample'],
                )
                print("feature_group_param: ", model)

                # scoring
                scoring = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score, average='macro'),
                    'recall': make_scorer(recall_score, average='macro'),
                    'f1': make_scorer(f1_score, average='macro'),
                    'roc_auc': make_scorer(roc_auc_score)
                }

                # cross validation
                cv_results = cross_validate(model, x_fit, y_fit, cv=cv, scoring=scoring, return_train_score=False,
                                            n_jobs=n_jobs)
                average_scores = {metric: scores.mean() for metric, scores in cv_results.items()}
                logging.info(f"Average scores: {average_scores}")
                result.append(average_scores)

        else:
            print(f"Feature group {idx + 1} is not available in the dataset.")
            continue

        del x_fit, y_fit, model, cv, cv_results, average_scores
        gc.collect()
    return result


if __name__ == '__main__':
    data_resample = joblib.load(File_resample_data)
    data_best_param = joblib.load(File_best_param)

    X_list, y_list = extract_features_targets(data_resample)
    feature_best_param = extract_best_params(data_best_param)

    # Call the function
    result = train_model(X_list, y_list, feature_best_param)
