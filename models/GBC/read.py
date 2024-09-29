from collections import Counter

import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split


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
    df['percentage_b'] = ((df['ended_D'] - df['created_D'].astype(float)) / df['created_D'].astype(float)) * 100
    df['percentage_b'] = ((df['ended_B'] - df['created_B']) / df['created_B']) * 100
    df['percentage_cp'] = ((df['ended_CP'] - df['created_CP']) / df['created_CP']) * 100
    df['percentage_c'] = ((df['ended_C'] - df['created_C']) / df['created_C']) * 100
    df['percentage_ooa'] = ((df['ended_OOA'] - df['created_OOA']) / df['created_OOA']) * 100
    return df


if __name__ == '__main__':

    df_seatunnel = pd.read_parquet('../output/seatunnel_prepare_to_train.parquet')
    df_ozone = pd.read_parquet('../output/ozone_prepare_to_train.parquet')
    df_pulsar = pd.read_parquet('../output/pulsar_prepare_to_train.parquet')

    X = percentage_smell(df_seatunnel)
    y = df_seatunnel['time_class']
    print('Original dataset shape %s' % Counter(y))

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    x_test, y_test, x_train, y_train = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=random_state)
