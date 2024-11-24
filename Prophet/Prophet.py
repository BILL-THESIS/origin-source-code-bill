import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

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


if __name__ == '__main__':
    seatunnal = pd.read_pickle('../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    pulsar = pd.read_pickle('../Sonar/output/tag_bug/pulsar_bug_comapare_time.pkl')
    ozone = pd.read_pickle('../Github/output/ozone_filtered_issue_bug.pkl')

    seatunnal = calculate_smell_bug(seatunnal)
    pulsar = calculate_smell_bug(pulsar)

    columns_y = ['diff_bug', 'diff_d', 'diff_b', 'diff_cp', 'diff_c', 'diff_ooa', 'diff_u']

    trend_each_smell_bug(pulsar, columns_y, 'Pulsar')
    trend_sum_smell(pulsar, 'sum_smell', 'Pulsar')

    # seatunnal_train = seatunnal[['completed_date', 'sum_smell']]
    # seatunnal_train.columns = ['ds', 'y']
    # seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)
    #
    # model = Prophet()
    # model.fit(seatunnal_train)
    #
    # future = model.make_future_dataframe(periods=365)
    # forecast = model.predict(future)
    #
    # model.plot(forecast)
    # model.plot_components(forecast)
    #
    # forecast_col = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    #
    # # รวมข้อมูลจริงเข้ากับ forecast
    # actual_and_predicted = pd.merge(seatunnal_train, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
    # print(actual_and_predicted.head())
    #
    #
    # plt.plot(actual_and_predicted['ds'], actual_and_predicted['y'], label='Actual')
    # plt.plot(actual_and_predicted['ds'], actual_and_predicted['yhat'], label='Forecast')
    # plt.fill_between(actual_and_predicted['ds'], actual_and_predicted['yhat_lower'], actual_and_predicted['yhat_upper'],
    #                  color='blue', alpha=0.2, label='Uncertainty Interval')
    # plt.legend()
    # plt.title('Seatunnal forecast sum smell compare with actual')
    # plt.xlabel('Date')
    # plt.ylabel('Sum smell')
    # plt.savefig('output/seatunnal_forecast_sum_smell_compare_actual.png')
    # plt.show()
    #
    # # Cross-validation
    # df_cv = cross_validation(model, period='365 days', horizon='90 days')
    # df_p = performance_metrics(df_cv)
    # print(df_p)


