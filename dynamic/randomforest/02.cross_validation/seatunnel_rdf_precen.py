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
    time_upper = data['total_hours'].sort_values(ascending=False)
    time_lower = data['total_hours'].sort_values(ascending=True)

    percen_10_upper = time_upper.iloc[int(0.1 * len(time_upper))]
    percen_10_lower = time_lower.iloc[int(0.1 * len(time_lower))]

    bins = [-float('inf'), percen_10_upper, float('inf')]
    labels = [0, 1]

    # bins = [-float('inf'), percen_10_lower, percen_10_upper, float('inf')]
    # labels = [0, 1, 2]
    data['time_category'] = pd.cut(data['total_hours'], bins=bins, labels=labels, right=False)
    return data, percen_10_lower, percen_10_upper

def split_data(data):
    X = data.drop(columns=['total_time', 'total_hours', 'time_category'])
    y = data['time_category']
    return X, y

def oversample_data(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def perform_grid_search(X_resampled, y_resampled):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [0.01, 0.1, 1],
        'min_samples_split': [0.01, 0.1, 1],
        'min_samples_leaf': np.arange(1, 9),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    min_samples_per_class = min(Counter(y_resampled).values())
    n_splits = min(10, min_samples_per_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring=scoring, refit='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_resampled, y_resampled)
    return grid_search, cv, rf, scoring

def evaluate_model(grid_search, X_resampled, y_resampled, cv, rf, scoring):
    best_rf = grid_search.best_estimator_
    cv_results_tunning = cross_validate(best_rf, X_resampled, y_resampled, cv=cv, scoring=scoring, return_train_score=False)
    cv_results_normal = cross_validate(rf, X_resampled, y_resampled, cv=cv, scoring=scoring, return_train_score=False)
    return cv_results_tunning, cv_results_normal, best_rf

def display_results(cv_results_tunning, cv_results_normal):
    average_scores_normal = {metric: scores.mean() for metric, scores in cv_results_normal.items()}
    average_scores_tunning = {metric: scores.mean() for metric, scores in cv_results_tunning.items()}
    for metric, score in average_scores_normal.items():
        print(f"{metric}_normal: {score:.4f}")
    for metric, score in average_scores_tunning.items():
        print(f"{metric}_tunning: {score:.4f}")

if __name__ == "__main__":
    data, data_group1_significant = load_data()
    data, q1, q3 = preprocess_data(data)
    X, y = split_data(data)
    X_resampled, y_resampled = oversample_data(X, y)
    grid_search, cv, rf, scoring = perform_grid_search(X_resampled, y_resampled)
    cv_results_tunning, cv_results_normal, best_rf= evaluate_model(grid_search, X_resampled, y_resampled, cv, rf, scoring)
    resulst = display_results(cv_results_tunning, cv_results_normal)