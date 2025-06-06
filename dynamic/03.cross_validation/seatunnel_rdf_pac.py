import os
import pickle
import logging
import pandas as pd
import gc
from collections import Counter
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from imblearn.over_sampling import SMOTE
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

OUTPUT_DIR = os.path.join("/dynamic/output/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

# File paths
INPUT_FILEPATH = os.path.join(OUTPUT_DIR, f"{project_name}_compare.pkl")
GROUP_FILEPATH = os.path.join(OUTPUT_DIR, f"{project_name}_correlation_group_13360.pkl")


def load_data(input_filepath=INPUT_FILEPATH, group_filepath=GROUP_FILEPATH):
    """Load dataset and feature groups."""
    logging.info("Loading data...")

    # Load feature groups
    with open(group_filepath, 'rb') as f:
        feature_groups = pickle.load(f)

    # Load dataset
    data = pd.read_pickle(input_filepath)

    # ลดขนาดของตัวแปรประเภทตัวเลข
    for col in data.select_dtypes(include=['int64']):
        data[col] = pd.to_numeric(data[col], downcast='integer')
    for col in data.select_dtypes(include=['float64']):
        data[col] = pd.to_numeric(data[col], downcast='float')

    logging.info("Data and feature groups loaded successfully.")
    return data, feature_groups


def preprocess_time_category(data):
    """Add time category column based on quantiles of total_hours."""
    logging.info("Processing time category...")

    # แปลง total_time เป็น timedelta และกำจัดค่า NaT
    data['total_time'] = pd.to_timedelta(data['total_time']).fillna(pd.Timedelta(0))

    # แปลงเวลาเป็นชั่วโมง
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600

    # คำนวณ quantile ที่ 75%
    q3 = data['total_hours'].quantile(0.75)

    # ตรวจสอบว่าค่า q3 มีค่ามากกว่า 0 หรือไม่
    if q3 > 0:
        data['time_category'] = pd.cut(
            data['total_hours'], bins=[-float('inf'), q3, float('inf')], labels=[0, 1], right=False
        )
    else:
        logging.warning("q3 is 0, setting all time_category to 0")
        data['time_category'] = 0

    # ลบคอลัมน์ที่ไม่จำเป็น
    data.drop(columns=['total_time', 'total_hours'], inplace=True)
    gc.collect()

    logging.info("Time category processed.")
    return data


if __name__ == "__main__":
    data, feature_groups = load_data()
    feature_groups = feature_groups[:5]
    data = preprocess_time_category(data)
    logging.info("Data preprocessing completed.")

    # Define target and features
    target = 'time_category'
    features = [col for col in data.columns if col != target]

    logging.info("Starting cross-validation...")

    n_jobs = 3

    result_df = pd.DataFrame()
    importance_features = []

    try:
        for x_feature in feature_groups:

            X = data[data.columns.intersection(x_feature)]
            X = X.fillna(0)
            y = data[target]

            # ตรวจสอบว่ามีหลาย class หรือไม่
            if len(set(y)) < 2:
                logging.warning("Skipping SMOTE due to a single class in the target variable.")
                continue

            # Oversample the minority class
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X, y = smote.fit_resample(X, y)

            logging.info(f"Processing feature group: {list(X.columns)}")

            model = RandomForestClassifier(random_state=42, n_jobs=n_jobs)

            facture_importance = model.fit(X, y).feature_importances_
            logging.info(f"Feature importance: {facture_importance}")
            importance_features.append({"feature_group": list(X.columns), "importance": facture_importance})

            param_grid = {
                'n_estimators': [50, 100, 200],
                'ccp_alpha': [0.01, 0.1, 1]
            }

            # Perform hyperparameter tuning
            grid_cv = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=n_jobs)
            grid_cv.fit(X, y)
            best_params = grid_cv.best_params_
            logging.info(f"Best Parameters Found: {best_params}")

            # Evaluate features using cross-validation
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            # ป้องกันกรณีที่ y มี class น้อยเกินไป
            min_class_samples = min(Counter(y).values())
            if min_class_samples < 2:
                logging.warning("Skipping cross-validation due to insufficient class samples.")
                continue

            cv = StratifiedKFold(n_splits=min(10, min_class_samples), shuffle=True, random_state=42)
            results = cross_validate(grid_cv.best_estimator_, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

            avg_results = {metric: scores.mean() for metric, scores in results.items()}
            avg_results['feature_group'] = list(X.columns)
            avg_results['best_params'] = best_params
            result_df = pd.concat([result_df, pd.DataFrame([avg_results])], ignore_index=True)

            logging.info(f"Feature group processed: Data preprocessing completed.")

            del X, y

        # Save results
        result_df.to_pickle(os.path.join(OUTPUT_DIR, f"{project_name}_results_{timestamp}.pkl"))
        with open(os.path.join(OUTPUT_DIR, f"{project_name}_importance_{timestamp}.pkl"), 'wb') as f:
            pickle.dump(importance_features, f)
        logging.info("Results saved successfully.")


    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
