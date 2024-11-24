import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import optuna


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


def optimize_prophet(df: pd.DataFrame, col: str, trials: int = 50) -> Prophet:
    """Use Optuna to optimize Prophet hyperparameters."""

    def objective(trial):
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True)
        holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True)
        changepoint_range = trial.suggest_float('changepoint_range', 0.8, 0.95)


        seatunnal_train = df[['completed_date', col]].rename(columns={'completed_date': 'ds', col: 'y'})
        seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)

        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_range=changepoint_range,
        )

        model.fit(seatunnal_train)

        # Perform cross-validation and calculate performance
        df_cv = cross_validation(model, period='365 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
        return df_p['rmse'].mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)

    best_params = study.best_params
    print("Best parameters:", best_params)

    # Train the final model with best parameters
    model = Prophet(
        seasonality_mode=best_params['seasonality_mode'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale'],
        changepoint_range=best_params['changepoint_range']
    )
    return model


def train_and_forecast(df: pd.DataFrame, col: str, model: Prophet):
    seatunnal_train = df[['completed_date', col]].rename(columns={'completed_date': 'ds', col: 'y'})
    seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)

    model.fit(seatunnal_train)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    plt.title(f'Forecast for {col}')
    plt.xlabel('Date')
    plt.ylabel(f'Y value of {col}')
    plt.savefig(f'output/forecast_{col}.png')
    plt.show()

    fig2 = model.plot_components(forecast)
    plt.title(f'Components for {col}')
    plt.savefig(f'output/forecast_{col}_components.png')
    plt.show()

    return forecast


if __name__ == '__main__':
    # Load data
    seatunnal = pd.read_pickle('../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    seatunnal = calculate_smell_bug(seatunnal)

    columns_y = ['diff_bug', 'diff_d', 'diff_b', 'diff_cp', 'diff_c', 'diff_ooa', 'diff_u', 'sum_smell']

    # Optimize Prophet model
    optimized_model = optimize_prophet(seatunnal, col='diff_bug', trials=50)

    # Train and forecast
    forecast = train_and_forecast(seatunnal, col='diff_bug', model=optimized_model)

    # Cross-validation
    df_cv = cross_validation(optimized_model, period='365 days', horizon='90 days')
    df_p = performance_metrics(df_cv)
    print(df_p)
