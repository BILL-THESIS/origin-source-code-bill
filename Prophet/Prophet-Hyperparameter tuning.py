import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import optuna
from textwrap import wrap

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

def optimize_prophet(df: pd.DataFrame, cols: list, trials: int = 100) -> dict:
    """Use Optuna to optimize Prophet hyperparameters for each column."""
    best_params_dict = {}

    def objective(trial, col):
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.01, 0.5, log=True)
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True)
        holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True)

        seatunnal_train = df[['completed_date', col]].rename(columns={'completed_date': 'ds', col: 'y'})
        seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)

        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
        )

        model.fit(seatunnal_train)

        # Perform cross-validation and calculate performance
        df_cv = cross_validation(model, period='365 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
        return df_p['rmse'].mean()

    for col in cols:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, col), n_trials=trials)
        best_params_dict[col] = study.best_params
        print(f"Best parameters for {col}:", study.best_params)

    return best_params_dict

def train_and_forecast(df: pd.DataFrame, cols: list, best_params_dict: dict, project_name: str) -> pd.DataFrame:
    for col in cols:
        best_params = best_params_dict[col]
        model = Prophet(
            seasonality_mode=best_params['seasonality_mode'],
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            holidays_prior_scale=best_params['holidays_prior_scale']
        )

        data_train = df[['completed_date', col]].rename(columns={'completed_date': 'ds', col: 'y'})
        data_train['ds'] = pd.to_datetime(data_train['ds']).dt.tz_localize(None)

        model.fit(data_train)

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        model.plot(forecast)
        plt.title("\n".join(wrap(f'{project_name} Forecast for {col} tuning.\nBest Params: {best_params}', 60)))
        plt.xlabel('Date')
        plt.ylabel(f'Y value of {col}')
        plt.tight_layout()
        plt.savefig(f'output/{project_name}_forecast_{col}_tuning.png')
        plt.show()

        model.plot_components(forecast)
        plt.title(f'{project_name} Forecast for {col} tuning.\nBest Params: {best_params}')
        plt.title("\n".join(wrap(f'{project_name} Forecast for {col} tuning.', 60)))
        plt.tight_layout()
        plt.savefig(f'output/{project_name}_forecast_{col}_components_tuning.png')
        # plt.show()

    return forecast

if __name__ == '__main__':
    # Load data
    seatunnal = pd.read_pickle('../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    pulsar = pd.read_pickle('../Sonar/output/tag_bug/pulsar_bug_comapare_time.pkl')
    pulsar_outlier = pd.read_pickle('../models/robust_outliers/output/pulsar_normal.pkl')

    seatunnal = calculate_smell_bug(seatunnal)
    pulsar_outlier = calculate_smell_bug(pulsar_outlier)
    pulsar = calculate_smell_bug(pulsar)


    columns_y = ['diff_bug', 'diff_d', 'diff_b', 'diff_cp', 'diff_c', 'diff_ooa', 'diff_u', 'sum_smell']

    # Optimize Prophet model
    optimized_model = optimize_prophet(pulsar, columns_y, trials=100)

    # suparate the positive and negative values
    check_value_positive = pulsar[pulsar['diff_bug'] > 0]
    check_value_negative = pulsar[pulsar['diff_bug'] < 0]
    check_value_equal = pulsar[pulsar['diff_bug'] == 0]


    # Train and forecast
    forecast_pulsar = train_and_forecast(pulsar, columns_y, best_params_dict=optimized_model, project_name='Pulsar')
    # forecast_seatunnal = train_and_forecast(seatunnal, columns_y, best_params_dict=optimized_model, project_name='Seatunnal')

    # cross-validation
    df_p_list = []
    summary_list = []
    for col in columns_y:
        model = Prophet(
            seasonality_mode=optimized_model[col]['seasonality_mode'],
            changepoint_prior_scale=optimized_model[col]['changepoint_prior_scale'],
            seasonality_prior_scale=optimized_model[col]['seasonality_prior_scale'],
            holidays_prior_scale=optimized_model[col]['holidays_prior_scale']
        )
        seatunnal_train = pulsar[['completed_date', col]].rename(columns={'completed_date': 'ds', col: 'y'})
        seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)
        model.fit(seatunnal_train)
        df_cv = cross_validation(model, period='365 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
        df_p['column'] = col
        # df_p_list.append(df_p.set_index('column'))
        df_p_list.append(df_p)

        # Calculate summary statistics for the evaluation metrics
        metrics = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
        summary_stats = df_p[metrics].describe()

        # Extract minimum values for comparison
        min_values = summary_stats.loc["min"]

        # Extract maximum values for comparison
        max_values = summary_stats.loc["max"]

        # Display summary statistics, minimum, and maximum values
        summary_list.append(summary_stats)

    # Verify the process
    print("Process verification completed.")