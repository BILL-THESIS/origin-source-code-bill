import pickle
from collections import Counter
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data():
    input_filepath = "../output/output/ozone_compare.pkl"
    data_group = pd.read_pickle("../output/output/ozone_correlation_main_group.pkl")
    data = pd.read_pickle(input_filepath)
    return data, data_group


def preprocess_data(data):
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    q1 = data['total_hours'].quantile(0.25)
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
            # X = data[group_i].fillna(0)
            X = pd.DataFrame(data[group_i].fillna(0))
            y = data_time['time_category']

            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)
            list_group_results.append((X_resampled, y_resampled))

    with open("ozone_resamples_each_smell.pkl", "wb") as file:
        pickle.dump(list_group_results, file)
