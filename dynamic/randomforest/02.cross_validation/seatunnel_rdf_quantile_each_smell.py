from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from imblearn.over_sampling import SMOTE
import pandas as pd

def load_data():
    input_filepath = "../../output/seatunnel_correlation.pkl"
    data_group1_significant = pd.read_pickle("../../output/seatunnel_correlation_group1_significant.pkl")
    data_group1_significant.fillna(0, inplace=True)
    data = pd.read_pickle(input_filepath)
    return data, data_group1_significant

def preprocess_data(data):
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    q1 = data['total_hours'].quantile(0.25)
    q3 = data['total_hours'].quantile(0.75)
    bins = [-float('inf'), q3, float('inf')]
    labels = [0, 1]
    data['time_category'] = pd.cut(data['total_hours'], bins=bins, labels=labels, right=False)
    return data, q1, q3

def split_data(data):
    drop_data = data.drop(columns=['total_time', 'total_hours', 'time_category'])

    list_data = []
    for i in drop_data:
        print(i)
        data_x = drop_data[i]
        data_y = data['time_category']
        list_data.append((data_x, data_y))
    return list_data

def process_data(list_x_y):
    list_mean_scores = []
    for x, y in list_x_y:
        print(f"===X===\n{x.name}\n===X===")
        print(f"===Y===\n{y.name}\n===Y===")

        X_resampled, y_resampled = oversample_data(x, y)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_resampled, y_resampled)

        average_scores_normal, average_scores_tunning = perform_cross_validation(X_resampled, y_resampled)
        list_mean_scores.append({
            'x': x.name,
            'accuracy': average_scores_normal['test_accuracy'],
            'precision': average_scores_normal['test_precision'],
            'recall': average_scores_normal['test_recall'],
            'f1': average_scores_normal['test_f1'],
            'accuracy_tunning': average_scores_tunning['test_accuracy'],
            'precision_tunning': average_scores_tunning['test_precision'],
            'recall_tunning': average_scores_tunning['test_recall'],
            'f1_tunning': average_scores_tunning['test_f1'],
            'Importance': model.feature_importances_
        })
    return list_mean_scores

def oversample_data(x, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    return X_resampled, y_resampled

def perform_cross_validation(X_resampled, y_resampled):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_resampled, y_resampled)
    min_samples_per_class = min(Counter(y_resampled).values())
    n_splits = min(10, min_samples_per_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [0.01, 0.1, 1],
        'min_samples_split': [0.01, 0.1, 1],
        'min_samples_leaf': [0.01, 0.1, 1],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring=scoring, refit='accuracy', n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_resampled, y_resampled)
    best_rf = grid_search.best_estimator_

    cv_results_normal = cross_validate(rf, X_resampled, y_resampled, cv=cv, scoring=scoring, return_train_score=False)
    cv_results_tunning = cross_validate(best_rf, X_resampled, y_resampled, cv=cv, scoring=scoring, return_train_score=False)
    average_scores_normal = {metric: scores.mean() for metric, scores in cv_results_normal.items() if 'test_' in metric}
    average_scores_tunning = {metric: scores.mean() for metric, scores in cv_results_tunning.items() if 'test_' in metric}
    for metric, score in average_scores_normal.items():
        print(f"{metric}: {score:.4f}")
    return average_scores_normal, average_scores_tunning

if __name__ == "__main__":
    data, data_group1_significant = load_data()
    data, q1, q3 = preprocess_data(data)
    list_x_y = split_data(data)
    list_mean_scores = process_data(list_x_y)
    df = pd.DataFrame(list_mean_scores)