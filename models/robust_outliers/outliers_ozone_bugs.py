import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_smell_bug(df: pd.DataFrame) -> pd.DataFrame:
    # rename_dict = {
    #     'Dispensables_created': 'created_d',
    #     'Bloaters_created': 'created_b',
    #     'Change Preventers_created': 'created_cp',
    #     'Couplers_created': 'created_c',
    #     'Object-Orientation Abusers_created': 'created_ooa',
    #     'Uncategorized_created': 'created_u',
    #     'Dispensables_ended': 'ended_d',
    #     'Bloaters_ended': 'ended_b',
    #     'Change Preventers_ended': 'ended_cp',
    #     'Couplers_ended': 'ended_c',
    #     'Object-Orientation Abusers_ended': 'ended_ooa',
    #     'Uncategorized_ended': 'ended_u'
    # }
    # df = df.rename(columns=rename_dict)
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
    # df['value_ended'] = pd.to_numeric(df['value_ended'], errors='coerce')
    # df['value_created'] = pd.to_numeric(df['value_created'], errors='coerce')
    # df['diff_bug'] = df['value_ended'] - df['value_created']
    # df['percentage_bug'] = ((df['value_ended'] - df['value_created']) / df['value_created']) * 100

    # for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
    #     df[f'diff_{col}'] = df[f'ended_{col}'] - df[f'created_{col}']
    #     df[f'percentage_{col}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

    # Add a new column for year-month
    df['year_month'] = pd.to_datetime(df['created_at']).dt.to_period('M')

    return df


def robust_outlier_detection(df: pd.DataFrame, threshold: float = 2.24) -> tuple:
    data = df['total_time']

    median = data.median()
    mad = np.abs(data - median).median()
    madn = mad / 0.6745

    low_outliers = (data - median) / madn < -threshold
    high_outliers = (data - median) / madn > threshold

    df_low_outliers = df[low_outliers]
    df_high_outliers = df[high_outliers]
    df_normal = df[~(low_outliers | high_outliers)]

    return df_normal, df_low_outliers, df_high_outliers


def plot_compare_shape(summary_df: pd.DataFrame, project_name: str):
    bar_width = 0.25
    plt.figure(figsize=(15, 8))

    x_labels = summary_df['Year_Month'].astype(str)  # Convert Period to string for plotting

    plt.bar(summary_df.index - bar_width, summary_df['Low outliers'], color='purple', width=bar_width,
            label='Low Outliers')
    plt.bar(summary_df.index, summary_df['Normal'], color='green', width=bar_width, label='Outliers Robust')
    plt.bar(summary_df.index + bar_width, summary_df['High outliers'], color='red', width=bar_width,
            label='High Outliers')

    plt.xticks(ticks=summary_df.index, labels=x_labels, rotation=45, ha='right')
    plt.title(f'{project_name} - Outlier Comparison by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'../../models/robust_outliers/output/{project_name}_outliers.png')
    plt.show()


if __name__ == '__main__':
    data_original = pd.read_pickle("../../Github/output/ozone_filtered_issue_bug.pkl")
    data_original = calculate_smell_bug(data_original)

    # Group by `year_month` instead of `year`
    grouped = data_original.groupby('year_month')
    results = {period: robust_outlier_detection(group) for period, group in grouped}

    summary_df = pd.DataFrame([
        {
            'Year_Month': period,
            'Normal': result[0].shape[0],
            'Low outliers': result[1].shape[0],
            'High outliers': result[2].shape[0]
        }
        for period, result in results.items()
    ])

    summary_df.reset_index(drop=True, inplace=True)
    plot_compare_shape(summary_df, 'ozone')
