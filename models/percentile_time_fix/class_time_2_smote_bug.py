import joblib
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from collections import Counter
from imblearn.over_sampling import SMOTE


# from sklearn.impute import SimpleImputer


def rename_columns(df):
    return df.rename(columns={
        'Dispensables_created': 'created_d',
        'Bloaters_created': 'created_b',
        'Change Preventers_created': 'created_cp',
        'Couplers_created': 'created_c',
        'Object-Orientation Abusers_created': 'created_ooa',
        'Uncategorized_created': 'created_u',
        'Dispensables_ended': 'ended_d',
        'Bloaters_ended': 'ended_b',
        'Change Preventers_ended': 'ended_cp',
        'Couplers_ended': 'ended_c',
        'Object-Orientation Abusers_ended': 'ended_ooa',
        'Uncategorized_ended': 'ended_u'
    })


def calculate_percentages(df):
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
    df['percentage_d'] = ((df['ended_d'] - df['created_d']) / df['created_d']) * 100
    df['percentage_b'] = ((df['ended_b'] - df['created_b']) / df['created_b']) * 100
    df['percentage_cp'] = ((df['ended_cp'] - df['created_cp']) / df['created_cp']) * 100
    df['percentage_c'] = ((df['ended_c'] - df['created_c']) / df['created_c']) * 100
    df['percentage_ooa'] = ((df['ended_ooa'] - df['created_ooa']) / df['created_ooa']) * 100
    return df


def process_data(df, percentile_df):
    results = []
    for _, row in percentile_df.iterrows():
        percentile_value = row[0]
        index_time01 = row.name
        df_point = df.copy()
        df_point['time_class'] = (df_point['total_time'] < percentile_value).apply(lambda x: 0 if x else 1)
        df_point['index_time01'] = [index_time01] * len(df_point)
        df_point['percentile_point_time01'] = [percentile_value] * len(df_point)
        results.append(df_point)
    return results


def check_amount_time_class(df_list):
    save_df_good = []
    save_df_bad = []
    for df in df_list:
        t_counts = df['time_class'].value_counts()
        t_0 = t_counts.get(0, 0)
        t_1 = t_counts.get(1, 0)

        if all(t >= 6 for t in [t_0, t_1]):
            save_df_good.append(df)
        else:
            save_df_bad.append(df)
    return save_df_good, save_df_bad


def split_data_x_y(df_list, random_state=3, test_size=0.3):
    metrics = {
        'precision_macro': [], 'recall_macro': [], 'f1_macro': [],
        'precision_smote_macro': [], 'recall_smote_macro': [], 'f1_smote_macro': [],
        'roc_auc_macro': [], 'accuracy': [], 'accuracy_smote': [],
        'y_original': [], 'y_resampled': [], 'y_train': [], 'y_train_smote': [],
        'time01': [], 'index_time01': []
    }

    for col in df_list:

        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=2)
        time_point = col['percentile_point_time01'].iloc[0]
        index_time = col['index_time01'].iloc[0]
        X = col[['created_d', 'created_b', 'created_cp', 'created_c', 'created_ooa',
                 'ended_d', 'ended_b', 'ended_cp', 'ended_c', 'ended_ooa',
                 'percentage_b', 'percentage_cp', 'percentage_c', 'percentage_ooa']]
        y = col['time_class']


        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, y_resampled,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)
        print('Original dataset shape %s' % Counter(y))
        print('Resampled dataset shape %s' % Counter(y_resampled))
        print('y_train dataset shape %s' % Counter(y_train))
        print('y_train_smote dataset shape %s' % Counter(y_train_smote))

        model = GradientBoostingClassifier()
        fit_normal = model.fit(X_train, y_train)
        fit_smote = model.fit(X_train_smote, y_train_smote)

        y_pred = cross_val_predict(fit_normal, X_train, y_train, cv=5)
        y_pred_smote = cross_val_predict(fit_smote, X_train_smote, y_train_smote, cv=5)

        metrics['accuracy'].append(accuracy_score(y_train, y_pred))
        metrics['accuracy_smote'].append(accuracy_score(y_train_smote, y_pred_smote))
        metrics['precision_macro'].append(precision_score(y_train, y_pred, average='macro'))
        metrics['recall_macro'].append(recall_score(y_train, y_pred, average='macro'))
        metrics['f1_macro'].append(f1_score(y_train, y_pred, average='macro'))
        metrics['precision_smote_macro'].append(precision_score(y_train_smote, y_pred_smote, average='macro'))
        metrics['recall_smote_macro'].append(recall_score(y_train_smote, y_pred_smote, average='macro'))
        metrics['f1_smote_macro'].append(f1_score(y_train_smote, y_pred_smote, average='macro'))
        metrics['roc_auc_macro'].append(roc_auc_score(y_train_smote, y_pred_smote, average='macro'))
        metrics['y_original'].append(Counter(y))
        metrics['y_resampled'].append(Counter(y_resampled))
        metrics['y_train'].append(Counter(y_train))
        metrics['y_train_smote'].append(Counter(y_train_smote))
        metrics['time01'].append(time_point)
        metrics['index_time01'].append(index_time)

    return metrics


def save_results(metrics, output_path):
    df_time_class2 = pd.DataFrame(metrics)
    with open(output_path, 'wb') as f:
        joblib.dump(df_time_class2, f)
    print("Save file done!")


if __name__ == '__main__':
    start_time = time.time()
    print(f"Start to normalize cluster at: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}")

    df_original = pd.read_pickle('../../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    df_original_rename = rename_columns(df_original)
    df_original_rename = calculate_percentages(df_original_rename)

    percentile_list = [np.percentile(df_original_rename['total_time'], range(1, 101))]
    percentile_df = pd.DataFrame(percentile_list).T

    all_df_list_time_class = process_data(df_original_rename, percentile_df)

    df_good, df_bad = check_amount_time_class(all_df_list_time_class)

    metrics = split_data_x_y(df_good)
    df_metrics = pd.DataFrame(metrics)

    data_50 = df_metrics[df_metrics['f1_macro'] <= 0.60]

    # save_results(metrics, '../output/seatunnel_bug_time_class2_smote.pkl')

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes, {total_time / 3600:.2f} hours)")
