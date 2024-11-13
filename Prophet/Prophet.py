import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly



def calculate_smell_bug(df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the percentage change for each smell type
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

    # Add a new column for year-month
    df['year'] = pd.to_datetime(df['completed_date']).dt.to_period('Y')
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')
    df['year_month_day'] = pd.to_datetime(df['completed_date']).dt.to_period('D')
    return df

if __name__ == '__main__':
    seatunnal = pd.read_pickle('../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    pulsar = pd.read_pickle('../Github/output/pulsar_filtered_issue_bug.pkl')
    ozone = pd.read_pickle('../Github/output/ozone_filtered_issue_bug.pkl')

    seatunnal = calculate_smell_bug(seatunnal)

    # prepare expected column names
    seatunnal_train = seatunnal[['completed_date', 'diff_bug']]
    seatunnal_train.columns = ['ds', 'y']
    seatunnal_train['ds'] = pd.to_datetime(seatunnal_train['ds']).dt.tz_localize(None)

    m = Prophet()
    m.fit(seatunnal_train)

    seatunnal_train.tail()

    future = m.make_future_dataframe(periods=200, include_history=True)
    future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = m.plot(forecast)
    plt.savefig('output/seatunnal_forecast.png')
    plt.show()

    fig2 = m.plot_components(forecast)
    plt.show()

    fig3 = plot_plotly(m, forecast)
    plt.show()

    fig4 = plot_components_plotly(m, forecast)
    plt.show()