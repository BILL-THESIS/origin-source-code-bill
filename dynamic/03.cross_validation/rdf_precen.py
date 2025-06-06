from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
import pandas as pd


if __name__ == "__main__":

    # File paths
    input_filepath = "../../output/seatunnel_correlation.pkl"
    data_group1_significant = pd.read_pickle("../../output/seatunnel_correlation_group1_significant.pkl")
    data_group1_significant.fillna(0, inplace=True)
    data = pd.read_pickle(input_filepath)

    # Convert 'total_time' to timedelta to calculate duration in hours
    data['total_time'] = pd.to_timedelta(data['total_time'])

    # Calculate total hours from the timedelta
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600

    q1 = data['total_hours'].quantile(0.25)
    q3 = data['total_hours'].quantile(0.75)

    # Bin the total_hours into categorical ranges
    # <= Q1, between Q1 and Q3, >= Q3
    bins = [-float('inf'), q1, q3, float('inf')]
    labels = [0, 1, 2]
    data['time_category'] = pd.cut(data['total_hours'], bins=bins, labels=labels, right=False)

    # Verify the binning results
    data['time_category'].value_counts(), q1, q3

    # แบ่งข้อมูลเป็น Features และ Target
    X = data.drop(columns=['total_time', 'total_hours', 'time_category'])
    # X = data_group1_significant
    y = data['time_category']
    print('Original dataset shape %s' % Counter(y))

    # Oversampling ด้วย SMOTE
    smote = SMOTE(k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_resampled))

    # Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # ตรวจสอบจำนวน fold ที่เหมาะสม
    min_samples_per_class = min(Counter(y_resampled).values())
    n_splits = min(10, min_samples_per_class)
    print(f"Using {n_splits} folds for cross-validation.")

    # 10-fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # กำหนด metric ต่างๆ ที่ต้องการ
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }

    # ใช้ Cross-Validation พร้อมรักษาสัดส่วน class
    cv_results = cross_validate(rf, X_resampled, y_resampled, cv=cv, scoring=scoring, return_train_score=False)


    # คำนวณค่าเฉลี่ยของแต่ละ metric
    average_scores = {metric: scores.mean() for metric, scores in cv_results.items() if 'test_' in metric}

    # แสดงผลลัพธ์
    for metric, score in average_scores.items():
        print(f"{metric}: {score:.4f}")
