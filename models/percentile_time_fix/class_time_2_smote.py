import pandas as pd
import numpy as np
import time

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from collections import Counter
from imblearn.over_sampling import SMOTE


def percentage_smell(df):
    df = df.rename(columns={'begin_Dispensables': 'created_D',
                            'begin_Bloaters': 'created_B',
                            'begin_Change Preventers': 'created_CP',
                            'begin_Couplers': 'created_C',
                            'begin_Object-Orientation Abusers': 'created_OOA',
                            'end_Dispensables': 'ended_D',
                            'end_Bloaters': 'ended_B',
                            'end_Change Preventers': 'ended_CP',
                            'end_Couplers': 'ended_C',
                            'end_Object-Orientation Abusers': 'ended_OOA'})

    df['percentage_b'] = ((df['ended_D'] - df['created_D'].astype(float)) / df['created_D'].astype(float)) * 100
    df['percentage_b'] = ((df['ended_B'] - df['created_B']) / df['created_B']) * 100
    df['percentage_cp'] = ((df['ended_CP'] - df['created_CP']) / df['created_CP']) * 100
    df['percentage_c'] = ((df['ended_C'] - df['created_C']) / df['created_C']) * 100
    df['percentage_ooa'] = ((df['ended_OOA'] - df['created_OOA']) / df['created_OOA']) * 100

    return df


def process_data(df, percentile_df):
    results = []

    for _, row in percentile_df.iterrows():
        # Assuming 'row' is a Series with one value
        percentile_value = row[0]
        index_time01 = row.name
        print("point ::", percentile_value)
        print("index_time01 ::", index_time01)

        # Create a copy of the DataFrame for the current percentile
        df_point = df.copy()

        # Apply the condition using vectorized operations
        df_point['time_class'] = (df_point['total_time'] < percentile_value).apply(lambda x: 0 if x else 1)
        df_point['index_time01'] = [index_time01] * len(df_point)
        df_point['percentile_point_time01'] = [percentile_value] * len(df_point)
        # Append the processed DataFrame to results
        results.append(df_point)

    return results


def split_data_x_y(df, random_state=3, test_size=0.3):
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []

    precision_smote_macro_list = []
    recall_smote_macro_list = []
    f1_smote_macro_list = []
    roc_auc_macro_list = []

    acc_list = []
    acc_smote_list = []

    y_original_list = []
    y_resampled_list = []
    y_train_list = []
    y_train_smote_list = []

    time01_list = []
    index_time01_list = []
    for col in df:
        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=2)
        time_point = col['percentile_point_time01'].iloc[0]
        index_time = col['index_time01'].iloc[0]
        X = col[['created_D', 'created_B', 'created_CP', 'created_C', 'created_OOA',
                 'ended_D', 'ended_B', 'ended_CP', 'ended_C', 'ended_OOA',
                 'percentage_b', 'percentage_cp', 'percentage_c', 'percentage_ooa']]
        y = col['time_class']

        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train_smote, X_test_smote, y_train_somte, y_test_smote = train_test_split(X_resampled, y_resampled,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)

        print('Original dataset shape %s' % Counter(y))
        print('Resampled dataset shape %s', Counter(y_resampled))
        print('y_train dataset shape %s', Counter(y_train))
        print('y_train dataset shape %s', Counter(y_train_somte))

        modle = GradientBoostingClassifier()
        fit_normal = modle.fit(X_train, y_train)
        fit_smote = modle.fit(X_train_smote, y_train_somte)

        y_pred = cross_val_predict(fit_normal, X_train, y_train, cv=5)
        y_pred_smote = cross_val_predict(fit_smote, X_train_smote, y_train_somte, cv=5)

        acc_normal = accuracy_score(y_train, y_pred)
        acc_smote = accuracy_score(y_train_somte, y_pred_smote)

        precision_macro_list.append(precision_score(y_train, y_pred, average='macro'))
        recall_macro_list.append(recall_score(y_train, y_pred, average='macro'))
        f1_macro_list.append(f1_score(y_train, y_pred, average='macro'))

        precision_smote_macro_list.append(precision_score(y_train_somte, y_pred_smote, average='macro'))
        recall_smote_macro_list.append(recall_score(y_train_somte, y_pred_smote, average='macro'))
        f1_smote_macro_list.append(f1_score(y_train_somte, y_pred_smote, average='macro'))
        roc_auc_macro_list.append(roc_auc_score(y_train_somte, y_pred_smote, average='macro'))

        acc_list.append(acc_normal)
        acc_smote_list.append(acc_smote)

        y_original_list.append(Counter(y))
        y_resampled_list.append(Counter(y_resampled))
        y_train_list.append(Counter(y_train))
        y_train_smote_list.append(Counter(y_train_somte))

        time01_list.append(time_point)
        index_time01_list.append(index_time)

    return (precision_macro_list, recall_macro_list, f1_macro_list,
            precision_smote_macro_list, recall_smote_macro_list, f1_smote_macro_list,
            roc_auc_macro_list, time01_list, index_time01_list,
            acc_list, acc_smote_list, y_original_list, y_resampled_list,
            y_train_list, y_train_smote_list)


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    df_original = pd.read_parquet('../output/seatunnel_all_information.parquet')

    # prepare the data X
    df_original_rename = percentage_smell(df_original)

    # prepare percentile to divide time class for 2 classes
    percentile_list = [np.percentile(df_original['total_time'], range(1, 100))]
    percentile_df = pd.DataFrame(percentile_list).T

    # conditions to split the data into 2 classes
    all_df_list_time_class = process_data(df_original_rename, percentile_df)

    (precision_macro_list, recall_macro_list, f1_macro_list,
     precision_smote_macro_list, recall_smote_macro_list, f1_smote_macro_list,
     roc_auc_macro_list, time01_list, index_time01_list,
     acc_list, acc_smote_list, y_original_list, y_resampled_list,
     y_train_list, y_train_smote_list) = split_data_x_y(all_df_list_time_class)

    df_time_class2 = pd.DataFrame({
        'index_time01': index_time01_list,
        'time01': time01_list,
        'accuracy': acc_list,
        'precision_macro': precision_macro_list,
        'recall_macro': recall_macro_list,
        'f1_macro': f1_macro_list,

        'accuracy_smote': acc_smote_list,
        'precision_smote_macro': precision_smote_macro_list,
        'recall_smote_macro': recall_smote_macro_list,
        'f1_smote_macro': f1_smote_macro_list,
        'roc_auc_macro': roc_auc_macro_list,

        'y_original': y_original_list,
        'y_resample': y_resampled_list,
        'y_train': y_train_list,
        'y_train_smote': y_train_smote_list,
    })

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
