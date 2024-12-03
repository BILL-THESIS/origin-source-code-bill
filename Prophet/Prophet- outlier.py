import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import RobustScaler

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

def trend_each_smell_bug(df: pd.DataFrame, columns_y: list, project_name: str):
    for col in columns_y:
        if 'completed_date' in df.columns and col in df.columns:
            seatunnal_train = df[['completed_date', col]]
        else:
            print(f"Columns 'completed_date' or '{col}' not found in seatunnal DataFrame")
            continue
        seatunnal_train.columns = ['ds', 'y']
        seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)

        m = Prophet()
        m.fit(seatunnal_train)

        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

        m.plot(forecast)
        plt.title(f'{project_name} forecast {col}')
        plt.xlabel('Date')
        plt.ylabel(f'Y value of {col}')
        plt.tight_layout()
        plt.savefig(f'output/{project_name}_forecast_{col}.png')
        plt.show()

        m.plot_components(forecast)
        plt.title(f'{project_name} forecast {col} components')
        plt.xlabel('Date')
        plt.ylabel(f'Y value of {col}')
        plt.tight_layout()
        plt.savefig(f'output/{project_name}_forecast_{col}_components.png')
        plt.show()


def trend_sum_smell(df: pd.DataFrame, col: str, project_name: str):
    seatunnal_train = df[['completed_date', col]]
    seatunnal_train.columns = ['ds', 'y']
    seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)

    m = Prophet()
    m.fit(seatunnal_train)

    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    m.plot(forecast)
    plt.title(f'{project_name} forecast {col}')
    plt.xlabel('Date')
    plt.ylabel(f'Y value of {col}')
    plt.tight_layout()
    plt.savefig(f'output/{project_name}_forecast_{col}.png')
    plt.show()

    m.plot_components(forecast)
    plt.title(f'{project_name} forecast {col} components')
    plt.xlabel('Date')
    plt.ylabel(f'Y value of {col}')
    plt.tight_layout()
    plt.savefig(f'output/{project_name}_forecast_{col}_components.png')
    plt.show()

def remove_outliers_robust(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    list_outliers = []
    for col in columns:
        scaler = RobustScaler()
        df[col] = scaler.fit_transform(df[col])
        list_outliers.append(df[col])
    return list_outliers


if __name__ == '__main__':

    pulsar = pd.read_pickle('../models/robust_outliers/output/pulsar_normal_diff_bug.pkl')

    # columns_y = ['diff_bug', 'diff_d', 'diff_b', 'diff_cp', 'diff_c', 'diff_ooa', 'diff_u']
    columns_y = ['diff_bug']

    pulsar = calculate_smell_bug(pulsar)

    # separate the positive and negative values
    check_value_positive = pulsar[pulsar['diff_bug'] > 0]
    check_value_negative = pulsar[pulsar['diff_bug'] < 0]
    check_value_equal = pulsar[pulsar['diff_bug'] == 0]

    # trend_each_smell_bug(check_value_negative, columns_y, 'Pulsar negative bug - not Tuning ')
    # # trend_sum_smell(pulsar, 'sum_smell', 'Pulsar')
    #
    # train = check_value_negative[['completed_date', 'diff_bug']]
    # train.columns = ['ds', 'y']
    # train['ds'] = pd.to_datetime(train['ds']).dt.tz_localize(None)
    #
    # model = Prophet()
    # model.fit(train)
    #
    # future = model.make_future_dataframe(periods=365)
    # forecast = model.predict(future)
    #
    # model.plot(forecast)
    # model.plot_components(forecast)
    #
    # forecast_col = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    #
    #
    # Cross-validation
    df_p_list_normal = []
    summary_list = []
    for col in columns_y:
        model = Prophet()
        train_df_p = check_value_negative[['completed_date', col]].rename(columns={'completed_date': 'ds', col: 'y'})
        train_df_p['ds'] = pd.to_datetime(train_df_p['ds']).dt.tz_localize(None)
        model.fit(train_df_p)
        df_cv = cross_validation(model, period='365 days', horizon='90 days')
        df_p = performance_metrics(df_cv)
        df_p['column'] = col
        # df_p_list.append(df_p.set_index('column'))
        df_p_list_normal.append(df_p)

        # Calculate summary statistics for the evaluation metrics
        metrics = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
        summary_stats = df_p[metrics].describe()

        # Extract minimum values for comparison
        min_values = summary_stats.loc["min"]

        # Extract maximum values for comparison
        max_values = summary_stats.loc["max"]

        # Display summary statistics, minimum, and maximum values
        summary_list.append(summary_stats)


