import pickle
import logging
import datetime
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
INPUT_FILEPATH = "../output/output/seatunnel_compare.pkl"
GROUP_FILEPATH = "../output/output/seatunnel_correlation_group_13360.pkl"
OUTPUT_DIR = "../output/"


def load_data(input_filepath=INPUT_FILEPATH, group_filepath=GROUP_FILEPATH):
    """Load dataset and feature groups."""
    logging.info("Loading data...")
    with open(group_filepath, 'rb') as f:
        feature_groups = pickle.load(f)
    data = pd.read_pickle(input_filepath)
    logging.info("Data and feature groups loaded successfully.")
    return data, feature_groups


def preprocess_time_category(data):
    """Add time category column based on quantiles of total_hours."""
    logging.info("Processing time category...")
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    q3 = data['total_hours'].quantile(0.75)
    data['time_category'] = (data['total_hours'] >= q3).astype(int)
    logging.info("Time category processed.")
    return data


def tune_hyperparameters(X, y, cv):
    """Perform hyperparameter tuning using GridSearchCV."""
    logging.info("Starting Grid Search for hyperparameter tuning...")
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    #     'max_features': ['sqrt'],
    #     'bootstrap': [True],
    #     'class_weight': ['balanced']
    # }

    # method1
    # param_grid = {
    #     'n_estimators': [50, 100, 200, 300],
    #     'max_depth': [0.001, 0.01, 0.1, 1],
    #     'min_samples_split': [1, 5, 10, 15, 25, 100],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False],
    #     'class_weight': ['balanced', 'balanced_subsample']
    # }

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.01, 0.1, 1],
        'max_depth': [1, 5, 10],
        'min_samples_split': [1, 5, 10],
        'min_impurity_decrease': [1, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'min_weight_fraction_leaf': [0.001, 0.01, 0.1, 1],
        'ccp_alpha': [0.001, 0.01, 0.1, 1]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    logging.info(f"Best Parameters Found: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_features(X, y, model, cv, scoring):
    """Evaluate features using cross-validation and compute feature importance."""
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    logging.info(f"Evaluating features: {list(X.columns)}")
    cv_results = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=scoring, n_jobs=-1)
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
    return results_df, importance_df


def main():
    """Main execution function."""
    data, feature_groups = load_data()
    data = preprocess_time_category(data)
    y = data['time_category']

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'roc_auc': make_scorer(roc_auc_score)
    }

    cv = StratifiedKFold(n_splits=min(10, min(Counter(y).values())), shuffle=True, random_state=42)

    list_results, list_importances = [], []

    with ThreadPoolExecutor(max_workers=8) as executor:
        feature_processing_tasks = [
            executor.submit(process_feature_group, fs, data, y, RandomForestClassifier(random_state=42, n_jobs=-1), cv,
                            scoring) for fs in feature_groups]
        for future in feature_processing_tasks:
            df_results, importance_df = future.result()
            list_results.append(df_results)
            list_importances.append(importance_df)

    final_results_df = pd.concat(list_results, ignore_index=True)
    feature_importances_df = pd.concat(list_importances, ignore_index=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_df.to_pickle(f"{OUTPUT_DIR}seatunnel_rdf_quantile_all_{timestamp}.pkl")
    feature_importances_df.to_pickle(f"{OUTPUT_DIR}seatunnel_feature_importances_{timestamp}.pkl")

    logging.info("Results saved successfully.")


if __name__ == "__main__":
    main()
