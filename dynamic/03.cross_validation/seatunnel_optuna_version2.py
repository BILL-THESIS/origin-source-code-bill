import os
import pickle
import logging
import pandas as pd
import gc
from collections import Counter
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from imblearn.over_sampling import SMOTE
from datetime import datetime
import lightgbm as lgb
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
project_name = "seatunnel"

OUTPUT_DIR = os.path.join("/dynamic/output/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.info(f"Running on project: {project_name}")

# File paths
INPUT_FILEPATH = os.path.join(OUTPUT_DIR, f"{project_name}_compare.pkl")
# GROUP_FILEPATH = os.path.join(OUTPUT_DIR, f"{project_name}_correlation_group_13360.pkl")
GROUP_FILEPATH = os.path.join(OUTPUT_DIR, f"/{OUTPUT_DIR}data_rank_all_group_sum_20.pkl")


def load_data(input_filepath=INPUT_FILEPATH, group_filepath=GROUP_FILEPATH):

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
    # สำหรับตัวอย่าง เราเลือกใช้เฉพาะ 5 กลุ่มคุณลักษณะแรก
    feature_groups = feature_groups[:5]
    data = preprocess_time_category(data)
    logging.info("Data preprocessing completed.")

    # กำหนด target และ features
    target = 'time_category'
    features = [col for col in data.columns if col != target]

    logging.info("Starting cross-validation with LightGBM and Optuna hyperparameter tuning...")

    n_jobs = 3

    result_df = pd.DataFrame()
    importance_features = []

    try:
        for x_feature in feature_groups:

            # เลือกเฉพาะคุณลักษณะที่อยู่ในกลุ่มนั้น
            X = data[data.columns.intersection(x_feature)]
            X = X.fillna(0)
            X_arry = X.to_numpy()
            y = data[target]
            y_arry = y.to_numpy()

            # ตรวจสอบว่ามีหลาย class หรือไม่
            if len(set(y)) < 2:
                logging.warning("Skipping SMOTE due to a single class in the target variable.")
                continue

            # Oversample the minority class ด้วย SMOTE
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X, y = smote.fit_resample(X, y)

            logging.info(f"Processing feature group: {list(X.columns)}")

            # ---------------------------------------------------------------------
            # 1. คำนวณ Feature Importance ด้วย LightGBM รุ่นเริ่มต้น
            # ---------------------------------------------------------------------
            default_model = lgb.LGBMClassifier(random_state=42, n_jobs=n_jobs)
            default_model.fit(X_arry, y_arry)
            feature_importance = default_model.feature_importances_
            logging.info(f"Feature importance: {feature_importance}")
            importance_features.append({"feature_group": list(X.columns), "importance": feature_importance})

            # ---------------------------------------------------------------------
            # 2. กำหนด cross-validation splits
            # ---------------------------------------------------------------------
            min_class_samples = min(Counter(y).values())
            if min_class_samples < 2:
                logging.warning("Skipping hyperparameter tuning and cross-validation due to insufficient class samples.")
                continue
            cv_splits = min(10, min_class_samples)
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

            # ---------------------------------------------------------------------
            # 3. กำหนด objective function สำหรับ Optuna
            # ---------------------------------------------------------------------
            def objective(trial):
                # กำหนด default parameters ทั้งหมดสำหรับ LightGBM
                default_params = {
                                "boosting_type": "gbdt",
                                "objective": "binary",  # ใช้สำหรับ binary classification
                                "n_estimators": 100,
                                "num_leaves": 31,
                                "max_depth": -1,
                                "learning_rate": 0.1,
                                "subsample": 1.0,
                                "subsample_freq": 0,
                                "colsample_bytree": 1.0,
                                "reg_alpha": 0.0,
                                "reg_lambda": 0.0,
                                "min_child_samples": 20,
                                "min_child_weight": 0.001,
                                "random_state": 42,
                                "n_jobs": n_jobs,
                                "verbose": -1
                }

                # อัปเดตค่า hyperparameter ที่จะทำการ tuning โดยใช้ optuna
                default_params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                default_params["num_leaves"] = trial.suggest_int("num_leaves", 20, 150)
                default_params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
                default_params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
                default_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
                default_params["subsample_freq"] = trial.suggest_int("subsample_freq", 0, 10)
                default_params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                default_params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 1.0)
                default_params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 1.0)
                default_params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
                default_params["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True)

                model = lgb.LGBMClassifier(**default_params)
                scores = cross_val_score(model, X_arry, y_arry, cv=cv, scoring="f1", n_jobs=n_jobs)
                return scores.mean()

            # ---------------------------------------------------------------------
            # 4. ทำ hyperparameter tuning ด้วย Optuna
            # ---------------------------------------------------------------------
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10, show_progress_bar=True)
            best_params = study.best_params
            logging.info(f"Best Parameters Found: {best_params}")

            # สร้าง dictionary สำหรับโมเดลที่ดีที่สุดโดยอิงจาก default แล้วอัปเดตด้วย best_params ที่ได้จาก Optuna
            best_model_params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_estimators": 100,
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "subsample_freq": 0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "min_child_samples": 20,
                "min_child_weight": 0.001,
                "random_state": 42,
                "n_jobs": n_jobs,
                "verbose": -1
            }
            # อัปเดตเฉพาะ parameter ที่ถูก tuning
            best_model_params.update(best_params)

            # ---------------------------------------------------------------------
            # 5. ประเมินประสิทธิภาพของโมเดลด้วย cross-validation โดยใช้ best_params
            # ---------------------------------------------------------------------
            best_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=n_jobs)
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='macro'),
                'recall': make_scorer(recall_score, average='macro'),
                'f1': make_scorer(f1_score, average='macro'),
                'roc_auc': make_scorer(roc_auc_score)
            }
            results = cross_validate(best_model, X_arry, y_arry, cv=cv, scoring=scoring, n_jobs=n_jobs)
            avg_results = {metric: scores.mean() for metric, scores in results.items()}
            avg_results['feature_group'] = list(X.columns)
            avg_results['best_params'] = best_params
            result_df = pd.concat([result_df, pd.DataFrame([avg_results])], ignore_index=True)

            logging.info("Feature group processed successfully.")

            del X, y
            del X_arry, y_arry
            gc.collect()

        # บันทึกผลลัพธ์
        result_filepath = os.path.join(OUTPUT_DIR, f"{project_name}_results_{timestamp}.pkl")
        importance_filepath = os.path.join(OUTPUT_DIR, f"{project_name}_importance_{timestamp}.pkl")
        result_df.to_pickle(result_filepath)
        with open(importance_filepath, 'wb') as f:
            pickle.dump(importance_features, f)
        logging.info("Results saved successfully.")

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
