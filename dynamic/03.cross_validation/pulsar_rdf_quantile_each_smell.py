from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from imblearn.over_sampling import SMOTE
import pandas as pd


def load_data():
    input_filepath = "../output/output/pulsar_compare.pkl"
    data_group = pd.read_pickle("../output/output/pulsar_correlation_main_group_7.pkl")
    data = pd.read_pickle(input_filepath)
    return data, data_group


def preprocess_data(data):
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    q3 = data['total_hours'].quantile(0.75)
    bins = [-float('inf'), q3, float('inf')]
    labels = [0, 1]
    data['time_category'] = pd.cut(data['total_hours'], bins=bins, labels=labels, right=False)
    return data


if __name__ == "__main__":
    data, data_group = load_data()
    data_time = preprocess_data(data)

    list_group_results = []

    for col in data_group:
        for group_i in col:
            X = data[group_i].fillna(0)
            y = data_time['time_category']

            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X.values.reshape(-1, 1), y.values.reshape(-1, 1))
            rf = RandomForestClassifier(random_state=42)

            min_samples_per_class = min(Counter(y_resampled).values())
            n_splits = min(10, min_samples_per_class)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='macro'),
                'recall': make_scorer(recall_score, average='macro'),
                'f1': make_scorer(f1_score, average='macro'),
                'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
            }
            cv_results_normal = cross_validate(rf, X_resampled, y_resampled, cv=cv, scoring=scoring, return_train_score=False)

            average_scores_normal = {metric: scores.mean() for metric, scores in cv_results_normal.items()}

            df_results = pd.DataFrame(average_scores_normal, index=[0])
            df_results['features'] = X.name
            list_group_results.append(df_results)

        final_results_df = pd.concat(list_group_results, ignore_index=True)
        print(final_results_df)

        final_results_df.to_pickle("../../output/pulsar_rdf_quantile_each_smell.pkl")