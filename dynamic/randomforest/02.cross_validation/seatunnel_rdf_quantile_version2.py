from collections import Counter
import pickle
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE

# ตั้งค่าการทำงานของ logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(input_filepath="../../output/seatunnel_compare.pkl",
              group_filepath="../../output/seatunnel_correlation_group_13360.pkl"):
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


def evaluate_features_with_importance(X, y, model, cv, scoring):
    """ประเมินผลการทำงานของ features และดึงค่า feature importances"""
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


if __name__ == "__main__":
    # โหลดข้อมูล
    data, feature_groups = load_data()
    data = preprocess_time_category(data)

    # กำหนดตัวแปรสำคัญ
    y = data['time_category']
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    rf = RandomForestClassifier(random_state=42)

    # เตรียม Cross-validation
    min_samples_per_class = min(Counter(y).values())
    n_splits = min(10, min_samples_per_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ประเมินผล features ทั้งหมด
    list_results = []
    list_importances = []
    for feature_set in feature_groups:
        X = data[list(feature_set)].fillna(0)
        average_scores, importance_df = evaluate_features_with_importance(X, y, rf, cv, scoring)

        # เก็บผลลัพธ์ใน DataFrame
        df_results = pd.DataFrame([average_scores])
        df_results['features'] = [list(X.columns)]
        list_results.append(df_results)

        # เก็บ feature importances
        importance_df['feature_set'] = str(list(X.columns))
        list_importances.append(importance_df)

    # รวมผลลัพธ์ทั้งหมด
    final_results_df = pd.concat(list_results, ignore_index=True)
    feature_importances_df = pd.concat(list_importances, ignore_index=True)

    logging.info("Evaluation complete. Saving results...")
    final_results_df.to_pickle("../../output/seatunnel_rdf_quantile_all.pkl")
    feature_importances_df.to_pickle("../../output/seatunnel_feature_importances_group.pkl")
    logging.info("Results and feature importances saved successfully.")
