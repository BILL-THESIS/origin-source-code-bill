import itertools
import os
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE


def check_time(df):
    df['merged_at'] = pd.to_datetime(df['merged_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
    return df[df['merged_at'].dt.year >= 2024], df[df['merged_at'].dt.year < 2024]


def percentage_smell(df):
    df = df.rename(columns={'created_Dispensables': 'created_D',
                            'created_Bloaters': 'created_B',
                            'created_Change Preventers': 'created_CP',
                            'created_Couplers': 'created_C',
                            'created_Object-Orientation Abusers': 'created_OOA',
                            'ended_Dispensables': 'ended_D',
                            'ended_Bloaters': 'ended_B',
                            'ended_Change Preventers': 'ended_CP',
                            'ended_Couplers': 'ended_C',
                            'ended_Object-Orientation Abusers': 'ended_OOA'})

    df['created_D'].astype(float)
    df['percentage_b'] = ((df['ended_D'] - df['created_D']) / df['created_D']) * 100
    df['percentage_b'] = ((df['ended_B'] - df['created_B']) / df['created_B']) * 100
    df['percentage_cp'] = ((df['ended_CP'] - df['created_CP']) / df['created_CP']) * 100
    df['percentage_c'] = ((df['ended_C'] - df['created_C']) / df['created_C']) * 100
    df['percentage_ooa'] = ((df['ended_OOA'] - df['created_OOA']) / df['created_OOA']) * 100
    return df


def calculate_percentiles(date_series):
    percentiles = np.percentile(date_series, range(0, 101))
    return pd.DataFrame(percentiles, columns=['percentile'])


def set_index_combinations_percentiles(percentile_df):
    return [percentile_df.iloc[list(comb)] for comb in itertools.combinations(range(len(percentile_df)), 2)]


def table_time_fix_percentile(percentile_dfs):
    time_points = []
    for df in percentile_dfs:
        time_01, time_12 = df['percentile']
        index_time01, index_time12 = df.index

        time_points.append({
            'index_time01': index_time01,
            'index_time12': index_time12,
            'time01': time_01,
            'time12': time_12
        })

    return pd.DataFrame(time_points)


def divide_time_class_2(df_original, df_time_point):
    results = []
    for index, row in df_time_point.iterrows():
        # Create a copy of the DataFrame for the current percentile
        time01 = row['time01']
        time12 = row['time12']
        time_fix_hours = df_original['total_time_hours']
        # time_fix_hours = df_original['total_time'].dt.total_seconds() / 3600

        values_time = []
        for time_i in time_fix_hours:
            if time_i <= time01:
                values_time.append(0)
                # print(f"time modify :: {time_i} < time01 :: {time01}")
            elif (time_i > time01) & (time_i < time12):
                values_time.append(1)
                # print(f"time01 :: {time01} >= time modify :: {time_i} < time12 :: {time12}")
            else:  # time_i >= time12
                values_time.append(2)
                # print(f"time modify :: {time_i} >=  time12 :: {time12}")

        # Create the 'time_class' column directly during iteration
        df_original['time_class'] = values_time
        df_original['index_time01'] = row['index_time01']
        df_original['time_01'] = row['time01']
        df_original['index_time12'] = row['index_time12']
        df_original['time_12'] = row['time12']

        # Append the modified DataFrame to results
        results.append(df_original.copy())
        # Avoid modifying the original DataFrame

    return results


def check_amount_time_class(dfs):
    save_df_good = []
    save_df_bad = []

    for df in dfs:
        class_counts = df['time_class'].value_counts()

        t_0 = class_counts.get(0, 0)
        t_1 = class_counts.get(1, 0)
        t_2 = class_counts.get(2, 0)

        # Check if all classes have at least 5 instances
        if t_0 >= 6 and t_1 >= 6 and t_2 >= 6:
            save_df_good.append(df)
        else:
            save_df_bad.append(df)

    return save_df_good, save_df_bad


def split_data_x_y(dfs, random_state=3, test_size=0.3):
    metrics = {
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_smote': [],
        'recall_smote': [],
        'f1_smote': [],
        'accuracy_normal': [],
        'accuracy_smote': [],
        'roc_auc_smote': [],
        'y_original': [],
        'y_resampled': [],
        'y_train_normal': [],
        'y_train_smote': [],
        'index_time01': [],
        'index_time12': [],
        'time01': [],
        'time12': [],
        'time0': [],
        'time1': [],
        'time2': []
    }

    for df in dfs:
        # Extract time-related columns
        metrics['index_time01'].append(df['index_time01'].iloc[0])
        metrics['index_time12'].append(df['index_time12'].iloc[0])
        metrics['time01'].append(df['time_01'].iloc[0])
        metrics['time12'].append(df['time_12'].iloc[0])

        # Feature matrix and target variable
        X = df[['created_D', 'created_B', 'created_CP', 'created_C', 'created_OOA',
                'ended_D', 'ended_B', 'ended_CP', 'ended_C', 'ended_OOA',
                'percentage_b', 'percentage_cp', 'percentage_c', 'percentage_ooa']]
        y = df['time_class']

        print(f'Original dataset shape: {Counter(y)}')

        # SMOTE resampling
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f'Resampled dataset shape: {Counter(y_resampled)}')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
            X_resampled, y_resampled, test_size=test_size, random_state=random_state
        )

        print(f'y_train dataset shape: {Counter(y_train)}')
        print(f'y_train_resampled dataset shape: {Counter(y_train_resampled)}')

        # Model training and evaluation
        model = GradientBoostingClassifier()
        normal_fit = model.fit(X_train, y_train)
        smote_fit = model.fit(X_train_resampled, y_train_resampled)

        y_pred_normal = cross_val_predict(normal_fit, X_train, y_train, cv=5)
        y_pred_smote = cross_val_predict(smote_fit, X_train_resampled, y_train_resampled, cv=5)

        # Metrics calculation
        metrics['precision_macro'].append(precision_score(y_train, y_pred_normal, average='macro'))
        metrics['recall_macro'].append(recall_score(y_train, y_pred_normal, average='macro'))
        metrics['f1_macro'].append(f1_score(y_train, y_pred_normal, average='macro'))
        metrics['accuracy_normal'].append(accuracy_score(y_train, y_pred_normal))

        metrics['precision_smote'].append(precision_score(y_train_resampled, y_pred_smote, average='macro'))
        metrics['recall_smote'].append(recall_score(y_train_resampled, y_pred_smote, average='macro'))
        metrics['f1_smote'].append(f1_score(y_train_resampled, y_pred_smote, average='macro'))
        metrics['accuracy_smote'].append(accuracy_score(y_train_resampled, y_pred_smote))

        y_pred_roc = smote_fit.predict_proba(X_test_resampled)
        metrics['roc_auc_smote'].append(roc_auc_score(y_test_resampled, y_pred_roc, multi_class='ovr'))

        # Store class distribution
        metrics['y_original'].append(Counter(y))
        metrics['y_resampled'].append(Counter(y_resampled))
        metrics['y_train_normal'].append(Counter(y_train))
        metrics['y_train_smote'].append(Counter(y_train_resampled))

        # Store time classes
        metrics['time0'].append((y == 0).sum())
        metrics['time1'].append((y == 1).sum())
        metrics['time2'].append((y == 2).sum())

    return (metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro'],
            metrics['precision_smote'], metrics['recall_smote'], metrics['f1_smote'],
            metrics['accuracy_normal'], metrics['accuracy_smote'],
            metrics['roc_auc_smote'],
            metrics['y_original'], metrics['y_resampled'], metrics['y_train_normal'], metrics['y_train_smote'],
            metrics['index_time01'], metrics['index_time12'], metrics['time01'], metrics['time12'],
            metrics['time0'], metrics['time1'], metrics['time2'])


if __name__ == '__main__':
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return joblib.load(f)


    ozone_api = pd.read_parquet('../../models/output/ozone_prepare_to_train.parquet')

    ozone_more_than_2024, ozone_less_than_2024 = check_time(ozone_api)
    ozone_less_than_2024 = percentage_smell(ozone_less_than_2024)

    ozone_percentiles = calculate_percentiles(ozone_less_than_2024['total_time_hours'])
    ozone_percentile_combinations = set_index_combinations_percentiles(ozone_percentiles)
    ozone_time_points = table_time_fix_percentile(ozone_percentile_combinations)
    ozone_time_classes = divide_time_class_2(ozone_less_than_2024, ozone_time_points)
    ozone_good, ozone_bad = check_amount_time_class(ozone_time_classes)

    (precision_macro, recall_macro, f1_macro,
     precision_smote, recall_smote, f1_smote,
     accuracy_normal, accuracy_smote,
     roc_auc_smote,
     y_original, y_resampled, y_train_normal, y_train_smote,
     index_time01, index_time12, time01, time12,
     time0, time1, time2) = split_data_x_y(ozone_good)

    ozone_time_class3_somte = pd.DataFrame({'accuracy': accuracy_normal,
                                         'precision_macro': precision_macro,
                                         'recall_macro': recall_macro,
                                         'f1_macro': f1_macro,

                                         'accuracy_smote': accuracy_smote,
                                         'precision_smote': precision_smote,
                                         'recall_smote': recall_smote,
                                         'f1_smote': f1_smote,
                                         'roc_auc_smote': roc_auc_smote,

                                         'y_original': y_original,
                                         'y_resample': y_resampled,
                                         'y_train': y_train_normal,
                                         'y_train_resample': y_train_smote,

                                         'index_time01': index_time01,
                                         'time01': time01,
                                         'index_time12': index_time12,
                                         'time12': time12,
                                         'time0': time0,
                                         'time1': time1,
                                         'time2': time2})


    with open('../../models/output/ozone_teat_model_time_class3_somte.parquet', 'wb') as f:
        joblib.dump(ozone_time_class3_somte, f)
        print("save file Done!")