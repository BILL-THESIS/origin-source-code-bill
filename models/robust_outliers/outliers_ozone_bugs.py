import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_total_time(df: pd.DataFrame) -> pd.DataFrame:
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])

    # Calculate years and months from total_time
    df['completed_date'] = pd.to_datetime(df['created_at']) + df['total_time']

    # Add a new column for year-month
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')

    return df


def robust_outlier_detection(df: pd.DataFrame, threshold: float = 2.24) -> tuple:
    data = df['total_time']

    median = data.median()
    mad = np.abs(data - median).median()
    madn = mad / 0.6745

    # Calculate lower and upper bounds for outliers
    low_outliers = (data - median) / madn < -threshold
    high_outliers = (data - median) / madn > threshold
    print("Lower  outlier: ", low_outliers)
    print("Upper outlier: ", high_outliers)
    print("\n")

    # Identify outliers based on the bounds
    outlier = (data - median).abs() / madn > threshold
    print("Sum outliers :", outlier.sum())

    df_low_outliers = df[low_outliers]
    df_high_outliers = df[high_outliers]
    df_normal = df[~(low_outliers | high_outliers)]

    print(
        f'Low outliers: {df_low_outliers.shape[0]} ,'
        f'High outliers: {df_high_outliers.shape[0]} '
        f'Normal: {df_normal.shape[0]}')

    return df_normal, df_low_outliers, df_high_outliers


def plot_compare_outlier(summary_df: pd.DataFrame, project_name: str):
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
    plt.savefig(f'../../models/robust_outliers/output/{project_name}_robits_outliers.png')
    plt.show()


if __name__ == '__main__':
    data_original = pd.read_pickle("../../Github/output/ozone_filtered_issue_bug.pkl")
    data_original = calculate_total_time(data_original)

    # Group by `year_month` instead of `year`
    grouped = data_original.groupby('year_month')
    normal, low_outliers, high_outliers = robust_outlier_detection(data_original)
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
    plot_compare_outlier(summary_df, 'ozone-total-time')
