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

OUTPUT_DIR = "/workspaces/origin-source-code-bill-1/dynamic/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

logging.info(f"running .... on ..... {project_name}")

# File paths
INPUT_FILEPATH = f"{OUTPUT_DIR}/{project_name}_compare.pkl"
GROUP_FILEPATH = f"{OUTPUT_DIR}/{project_name}_correlation_group.pkl"
OUTPUT_DIR = f"{OUTPUT_DIR}"


def load_data(input_filepath=INPUT_FILEPATH, group_filepath=GROUP_FILEPATH):
    """Load dataset and feature groups."""
    logging.info("Loading data...")
    with open(group_filepath, 'rb') as f:
        feature_groups = pickle.load(f)
    data = pd.read_pickle(input_filepath)

    # ลดขนาดของตัวแปรประเภทตัวเลข
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='float')

    logging.info("Data and feature groups loaded successfully.")
    return data, feature_groups


def preprocess_time_category(data):
    """Add time category column based on quantiles of total_hours."""
    logging.info("Processing time category...")
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    q3 = data['total_hours'].quantile(0.75)
    data['time_category'] = pd.cut(data['total_hours'], bins=[-float('inf'), q3, float('inf')], labels=[0, 1],
                                   right=False)
    # ลบทิ้งตัวแปรที่ไม่จำเป็นเพื่อประหยัดหน่วยความจำ
    data.drop(columns=['total_time', 'total_hours'], inplace=True)
    gc.collect()

    logging.info("Time category processed.")
    return data


def tune_hyperparameters(X, y, cv):
    """Perform hyperparameter tuning using GridSearchCV."""
    logging.info("Starting Grid Search for hyperparameter tuning...")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=4)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1_macro', n_jobs=4, verbose=1)
    grid_search.fit(X, y)
    logging.info(f"Best Parameters Found: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_features(X, y, model, cv, scoring):
    """Evaluate features using cross-validation and compute feature importance."""
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    logging.info(f"Evaluating features: {list(X.columns)}")
    cv_results = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=scoring, n_jobs=4)
    model.fit(X_resampled, y_resampled)
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    return {metric: scores.mean() for metric, scores in cv_results.items()}, importance_df


def process_feature_group(feature_set, data, y, model, cv, scoring):
    """Process a single feature group."""
    X = data[list(feature_set)].fillna(0)
    avg_scores, importance_df = evaluate_features(X, y, model, cv, scoring)
    results_df = pd.DataFrame([avg_scores])
    results_df['features'] = [list(X.columns)]
    importance_df['feature_set'] = str(list(X.columns))
    # ลบ DataFrame ที่ไม่ใช้แล้ว
    del X
    gc.collect()
    return results_df, importance_df


def process_feature_set(feature_set, data, y, best_model, cv, scoring):
    """ฟังก์ชันสำหรับประมวลผลแต่ละชุด feature"""
    return process_feature_group(feature_set, data, y, best_model, cv, scoring)


def main():
    """Main execution function."""
    data, feature_groups = load_data()
    # feature_groups = feature_groups[:20]
    data = preprocess_time_category(data)
    y = data['time_category']

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=1),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'roc_auc': make_scorer(roc_auc_score)
    }

    cv = StratifiedKFold(n_splits=min(10, min(Counter(y).values())), shuffle=True, random_state=42)

    # ทำ Oversampling เพียงครั้งเดียว
    X_all = data[list(set().union(*feature_groups))].copy()
    X_all.fillna(0, inplace=True)

    best_model, best_params = tune_hyperparameters(X_all, y, cv)

    # ลบ DataFrame ขนาดใหญ่ที่ไม่ใช้แล้ว
    del X_all
    gc.collect()

    list_results, list_importances = [], []

    logging.info("Starting parallel processing...")

    results = Parallel(n_jobs=4, backend="loky")(
        delayed(process_feature_set)(feature_set, data, y, best_model, cv, scoring) for feature_set in feature_groups
    )

    # แยกผลลัพธ์ออกเป็น list ต่าง ๆ
    for df_results, importance_df in results:
        list_results.append(df_results)
        list_importances.append(importance_df)

    # รวมผลลัพธ์เป็น DataFrame
    final_results_df = pd.concat(list_results, ignore_index=True)
    feature_importances_df = pd.concat(list_importances, ignore_index=True)
    best_params_df = pd.DataFrame([best_params])

    logging.info("Parallel processing complete. Saving results...")

    final_results_df.to_pickle(f"{OUTPUT_DIR}{project_name}_rdf_quantile_all_{timestamp}.pkl")
    feature_importances_df.to_pickle(f"{OUTPUT_DIR}{project_name}_feature_importances_{timestamp}.pkl")
    best_params_df.to_pickle(f"{OUTPUT_DIR}{project_name}_best_params_{timestamp}.pkl")

    logging.info("Results saved successfully.")


if __name__ == "__main__":
    main()