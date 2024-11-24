import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from textwrap import wrap

from keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def calculate_smell_bug(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {
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
    }

    df = df.rename(columns=rename_dict)
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
    df['completed_date'] = pd.to_datetime(df['created_at']) + df['total_time']
    df['value_ended'] = pd.to_numeric(df['value_ended'], errors='coerce')
    df['value_created'] = pd.to_numeric(df['value_created'], errors='coerce')
    df['diff_bug'] = df['value_ended'] - df['value_created']
    df['percentage_bug'] = ((df['value_ended'] - df['value_created']) / df['value_created']) * 100

    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'diff_{col}'] = df[f'ended_{col}'] - df[f'created_{col}']
        df[f'percentage_{col}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

    df['year'] = pd.to_datetime(df['completed_date']).dt.to_period('Y')
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')
    df['year_month_day'] = pd.to_datetime(df['completed_date']).dt.to_period('D')

    df['sum_smell'] = df['diff_d'] + df['diff_b'] + df['diff_cp'] + df['diff_c'] + df['diff_ooa'] + df['diff_u']
    return df

def optimize_lstm(df: pd.DataFrame, cols: list, trials: int = 100) -> dict:
    """Use Optuna to optimize LSTM hyperparameters for each column."""
    best_params_dict = {}

    def objective(trial, col):
        n_units = trial.suggest_int('n_units', 10, 100)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        epochs = trial.suggest_int('epochs', 10, 100)
        batch_size = trial.suggest_int('batch_size', 16, 128)

        data_train = df[col].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_train = scaler.fit_transform(data_train)

        X, y = [], []
        for i in range(60, len(data_train)):
            X.append(data_train[i-60:i, 0])
            y.append(data_train[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        for _ in range(n_layers):
            model.add(LSTM(units=n_units, return_sequences=True))
            model.add(Dropout(dropout))
        model.add(LSTM(units=n_units))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        loss = model.evaluate(X, y, verbose=0)
        return loss

    for col in cols:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, col), n_trials=trials)
        best_params_dict[col] = study.best_params
        print(f"Best parameters for {col}:", study.best_params)

    return best_params_dict

def train_and_forecast(df: pd.DataFrame, cols: list, best_params_dict: dict, project_name: str) -> pd.DataFrame:
    forecasts = {}
    for col in cols:
        best_params = best_params_dict[col]
        n_units = best_params['n_units']
        n_layers = best_params['n_layers']
        dropout = best_params['dropout']
        epochs = best_params['epochs']
        batch_size = best_params['batch_size']

        data_train = df[col].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_train = scaler.fit_transform(data_train)

        X, y = [], []
        for i in range(60, len(data_train)):
            X.append(data_train[i-60:i, 0])
            y.append(data_train[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        for _ in range(n_layers):
            model.add(LSTM(units=n_units, return_sequences=True))
            model.add(Dropout(dropout))
        model.add(LSTM(units=n_units))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        forecast = model.predict(X[-1].reshape(1, 60, 1))
        forecasts[col] = scaler.inverse_transform(forecast).flatten()

        plt.figure()
        plt.plot(scaler.inverse_transform(data_train), label='Observed')
        plt.plot(range(len(data_train), len(data_train) + len(forecast)), forecast, label='Forecast')
        plt.title("\n".join(wrap(f'{project_name} Forecast for {col} tuning.\nBest Params: {best_params}', 60)))
        plt.xlabel('Date')
        plt.ylabel(f'Y value of {col}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'output/{project_name}_forecast_{col}_tuning.png')
        plt.show()

    return forecasts

if __name__ == '__main__':
    # Load data
    seatunnal = pd.read_pickle('../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    pulsar = pd.read_pickle('../Sonar/output/tag_bug/pulsar_bug_comapare_time.pkl')
    pulsar_outlier = pd.read_pickle('../models/robust_outliers/output/pulsar_normal.pkl')

    seatunnal = calculate_smell_bug(seatunnal)
    pulsar_outlier = calculate_smell_bug(pulsar_outlier)

    columns_y = ['diff_bug', 'sum_smell']
    # columns_y = ['diff_bug', 'diff_d', 'diff_b', 'diff_cp', 'diff_c', 'diff_ooa', 'diff_u', 'sum_smell']


    # Optimize LSTM model
    optimized_model = optimize_lstm(pulsar_outlier, columns_y, trials=100)

    # Train and forecast
    forecast_pulsar = train_and_forecast(pulsar_outlier, columns_y, best_params_dict=optimized_model,
                                         project_name='Pulsar Outlier - LSTM')
    # Verify the process
    print("Process verification completed.")