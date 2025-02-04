import os
from collections import Counter
import pickle
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from imblearn.over_sampling import SMOTE
from datetime import datetime
from joblib import Parallel, delayed
from multiprocessing import Pool
# ใช้สำหรับ Parallel Processing
import gc

# ตั้งค่าการทำงานของ logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = "/workspaces/origin-source-code-bill-1/dynamic/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_data(input_filepath=f"{OUTPUT_DIR}/seatunnel_compare.pkl",
              group_filepath=f"{OUTPUT_DIR}/seatunnel_correlation_group_13360.pkl"):
    """โหลดข้อมูล input และกลุ่มของ features"""
    logging.info("Loading data...")
    with open(group_filepath, 'rb') as f:
        feature_groups = pickle.load(f)
    data = pd.read_pickle(input_filepath)
    logging.info("Data and feature groups loaded successfully.")
    return data, feature_groups


def preprocess_time_category(data):
    """เพิ่มคอลัมน์ time_category ตาม quantile ของ total_hours"""
    logging.info("Processing time category...")
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    q3 = data['total_hours'].quantile(0.75)
    bins = [-float('inf'), q3, float('inf')]
    labels = [0, 1]
    data['time_category'] = pd.cut(data['total_hours'], bins=bins, labels=labels, right=False)
    logging.info("Time category processed.")
    return data


def tune_hyperparameters(X, y, cv):
    """ทำ Hyperparameter tuning โดยใช้ GridSearchCV"""
    logging.info(f"Tuning hyperparameters for {X.shape[1]} features...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_features_with_importance(X, y, model, cv, scoring):
    """ประเมินผลโมเดล พร้อมคำนวณ feature importance"""
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    logging.info(f"Evaluating features: {list(X.columns)}")

    # Cross-validation
    cv_results = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=scoring, n_jobs=-1,
                                return_train_score=False)
    average_scores = {metric: scores.mean() for metric, scores in cv_results.items()}

    # Train model on resampled data to get feature importances
    model.fit(X_resampled, y_resampled)
    feature_importances = model.feature_importances_

    # จัดเก็บความสำคัญของฟีเจอร์ในรูป DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    return average_scores, importance_df


def process_feature_set(feature_set, data, y, cv, scoring):
    """รัน GridSearchCV และประเมินโมเดลสำหรับแต่ละชุดฟีเจอร์"""
    X = data[list(feature_set)].fillna(0)

    # 1. Hyperparameter tuning
    best_model = tune_hyperparameters(X, y, cv)

    # 2. ประเมินโมเดล
    average_scores, importance_df = evaluate_features_with_importance(X, y, best_model, cv, scoring)

    # 3. เก็บผลลัพธ์ใน DataFrame
    df_results = pd.DataFrame([average_scores])
    df_results['features'] = [list(X.columns)]

    # 4. เก็บ feature importances
    importance_df['feature_set'] = str(list(X.columns))

    # 5. เก็บ Best Model Parameters
    best_params = {'params': best_model.get_params(), 'features': list(X.columns)}

    del X  # ลบตัวแปร
    gc.collect()

    return df_results, importance_df, best_params


if __name__ == "__main__":
    # โหลดข้อมูล
    data, feature_groups = load_data()
    # feature_groups = feature_groups[:3]
    data = preprocess_time_category(data)

    # กำหนดตัวแปรสำคัญ
    y = data['time_category']
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=1),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'roc_auc': make_scorer(roc_auc_score)
    }

    # เตรียม Cross-validation
    min_samples_per_class = min(Counter(y).values())
    n_splits = min(10, min_samples_per_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # รัน GridSearchCV และประเมินฟีเจอร์แต่ละชุดแบบ Parallel Processing
    num_cores = 14  # ใช้ทุกคอร์ที่มีอยู่

    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(process_feature_set)(feature_set, data, y, cv, scoring) for feature_set in feature_groups
    )

    # แยกผลลัพธ์ออกมา
    list_results, list_importances, list_best_model = zip(*results)

    # รวมผลลัพธ์ทั้งหมด
    final_results_df = pd.concat(list_results, ignore_index=True)
    feature_importances_df = pd.concat(list_importances, ignore_index=True)
    best_params_df = pd.DataFrame(list_best_model)

    logging.info("Parallel processing complete. Saving results...")
    final_results_df.to_pickle(f"{OUTPUT_DIR}/seatunnel_rdf_quantile_all_{timestamp}.pkl")
    feature_importances_df.to_pickle(f"{OUTPUT_DIR}/seatunnel_feature_importances_{timestamp}.pkl")
    best_params_df.to_pickle(f"{OUTPUT_DIR}/seatunnel_best_params_{timestamp}.pkl")
    logging.info("Results and feature importances saved successfully.")
