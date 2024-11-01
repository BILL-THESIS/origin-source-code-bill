import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from bokeh.core.property.vectorization import value


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
    df['value_ended'] = pd.to_numeric(df['value_ended'], errors='coerce')
    df['value_created'] = pd.to_numeric(df['value_created'], errors='coerce')
    df['diff_bug'] = df['value_ended'] - df['value_created']
    df['percentage_bug'] = ((df['value_ended'] - df['value_created']) / df['value_created']) * 100

    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'diff_{col}'] = df[f'ended_{col}'] - df[f'created_{col}']
        df[f'percentage_{col}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

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

    print(f"Column: {data.name}")
    print(f"Median: {median}, MAD: {mad}, MADN: {madn}")
    print(f"Threshold: {threshold}, Low outliers detected: {low_outliers.sum()}, High outliers detected: {high_outliers.sum()}")

    return df_normal, df_low_outliers, df_high_outliers
def plot_compare_shape(summary_df: pd.DataFrame, project_name: str):
    bar_width = 0.25
    plt.figure(figsize=(12, 8))

    plt.bar(summary_df['Year'] - bar_width, summary_df['Low outliers'], color='purple', width=bar_width, label='Low Outliers')
    plt.bar(summary_df['Year'], summary_df['Normal'], color='green', width=bar_width, label='Outliers Robust')
    plt.bar(summary_df['Year'] + bar_width, summary_df['High outliers'], color='red', width=bar_width, label='High Outliers')

    for i in range(len(summary_df)):
        plt.text(summary_df['Year'][i] - bar_width, summary_df['Low outliers'][i], str(summary_df['Low outliers'][i]), ha='center', va='bottom')
        plt.text(summary_df['Year'][i], summary_df['Normal'][i], str(summary_df['Normal'][i]), ha='center', va='bottom')
        plt.text(summary_df['Year'][i] + bar_width, summary_df['High outliers'][i], str(summary_df['High outliers'][i]), ha='center', va='bottom')

    plt.title(f'{project_name} - Outlier Comparison by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(summary_df['Year'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'output/{project_name}_outlier_comparison.png')
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd


def plot_compare_shape_histogram(summary_df: pd.DataFrame, project_name: str):
    # Set up the figure size
    plt.figure(figsize=(12, 8))

    # Define bar width
    bar_width = 0.5
    years = summary_df['Year']

    # Create stacked bars for low outliers, normal, and high outliers
    plt.bar(years, summary_df['Low outliers'], color='purple', width=bar_width, label='Low Outliers')
    plt.bar(years, summary_df['Normal'], bottom=summary_df['Low outliers'], color='green', width=bar_width,
            label='Normal')
    plt.bar(years, summary_df['High outliers'], bottom=summary_df['Low outliers'] + summary_df['Normal'], color='red',
            width=bar_width, label='High Outliers')

    # Add count labels on the stacked bars
    for i in range(len(summary_df)):
        # Extract values for each category
        low_outliers = summary_df['Low outliers'][i]
        normal = summary_df['Normal'][i]
        high_outliers = summary_df['High outliers'][i]

        # Position labels in the center of each stacked section
        plt.text(years[i], low_outliers / 2, str(low_outliers), ha='center', va='center', color='white',
                 fontweight='bold', fontsize=9)
        plt.text(years[i], low_outliers + normal / 2, str(normal), ha='center', va='center', color='white',
                 fontweight='bold', fontsize=9)
        plt.text(years[i], low_outliers + normal + high_outliers / 2, str(high_outliers), ha='center', va='center',
                 color='white', fontweight='bold', fontsize=9)

    # Add titles, labels, and grid for clarity
    plt.title(f'{project_name} - Outlier Distribution by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(years, fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_original = pd.read_pickle('../../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    data_original = calculate_smell_bug(data_original)
    data_original['year'] = pd.to_datetime(data_original['created_at']).dt.year

    grouped = data_original.groupby('year')
    results = {year: robust_outlier_detection(group) for year, group in grouped}

    for year, result in results.items():
        print(f"Year: {year}")
        print(f"Normal: {result[0].shape[0]}, Low outliers: {result[1].shape[0]}, High outliers: {result[2].shape[0]}")

    summary_df = pd.DataFrame([
        {
            'Year': year,
            'Normal': result[0].shape[0],
            'Low outliers': result[1].shape[0],
            'High outliers': result[2].shape[0]
        }
        for year, result in results.items()
    ])

    plot_compare_shape(summary_df, 'seatunnal')