import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics


def calculate_smell_bug(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['normal', 'bug', 'vulnerability']:
        df[f'percentage_{col}'] = ((df[f'smell_{col}_ended'] - df[f'smell_{col}_created']) / df[
            f'smell_{col}_created']) * 100

    df['year'] = pd.to_datetime(df['completed_date']).dt.to_period('Y')
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')
    df['year_month_day'] = pd.to_datetime(df['completed_date']).dt.to_period('D')
    return df


def trend_each_smell_bug(df: pd.DataFrame, columns_y: list, name: str):
    df_p_list = []
    summary_list = []
    metrics = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]

    for col in columns_y:
        if 'completed_date' in df.columns and col in df.columns:
            train = df[['completed_date', col]]
        else:
            print(f"Columns 'completed_date' or '{col}' not found in DataFrame")
            continue

        train.columns = ['ds', 'y']
        train['ds'] = pd.to_datetime(train['ds']).dt.tz_localize(None)
        train.replace([np.inf, -np.inf], np.nan, inplace=True)
        train.dropna(subset=['y'], inplace=True)


        model = Prophet()
        model.fit(train)

        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        df_cv = cross_validation(model, period='60 days', horizon='60 days')
        df_p = performance_metrics(df_cv)
        df_p['column'] = col
        df_p_list.append(df_p)

        # Calculate summary statistics for the evaluation metrics
        summary_stats = df_p[metrics].describe()
        summary_stats['column'] = col
        summary_list.append(summary_stats)

        model.plot(forecast)
        plt.title(f'{name} forecast {col}')
        plt.xlabel('Date')
        plt.ylabel(f'Y value of {col}')
        plt.tight_layout()
        plt.savefig(f'output/{name}_forecast_{col}.png')
        plt.show()

        model.plot_components(forecast)
        plt.title(f'{name} forecast {col} components')
        plt.xlabel('Date')
        plt.ylabel(f'Y value of {col}')
        plt.tight_layout()
        plt.savefig(f'output/{name}_forecast_{col}_components.png')
        plt.show()

    return df_p_list, summary_list


if __name__ == '__main__':
    data = pd.read_pickle('../models/output/data_prepare_models/seatunnel_compare_types_smells.pkl')

    columns_y = ['smell_normal_created', 'smell_normal_ended', 'smell_bug_created',
                 'smell_bug_ended', 'smell_vulnerability_created', 'smell_vulnerability_ended',
                 'diff_normal', 'diff_bug', 'diff_vulnerability',
                 'percentage_normal', 'percentage_bug', 'percentage_vulnerability', 'total_time']

    data = calculate_smell_bug(data)

    df_p, summary_df_p = trend_each_smell_bug(data, columns_y, 'seatunnel')
