import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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

    # Add a new column for year-month
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')

    return df


def custom_time_hour_clustering_year(df: pd.DataFrame) -> pd.DataFrame:
    # Define the bin edges and labels for months
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    labels = list(range(12))

    # Extract month from the 'merged_at' column
    df['month'] = pd.to_datetime(df['completed_date']).dt.month
    df['year'] = pd.to_datetime(df['completed_date']).dt.year

    # Use pd.cut to assign values to the appropriate bins
    df['time_month_class'] = pd.cut(df['month'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def custom_time_clustering_media_average(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate median and average time
    median_time = df['total_time'].median()
    average_time = df['total_time'].mean()

    # Assign median and average time to the DataFrame
    df['time_class_median'] = df['total_time'].apply(lambda x: 0 if x > median_time else 1 if x < median_time else 2)
    df['time_class_average'] = df['total_time'].apply(lambda x: 0 if x > average_time else 1 if x < average_time else 2)
    return df


def separate_smell_integer(df):
    # Check the number of rows for each time_month_class
    positive, negative = {}, {}
    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        positive_list = df[df[f'diff_{col.lower()}'] >= 0].drop(
            columns=[f'diff_{c}' for c in ['d', 'b', 'cp', 'c', 'ooa', 'u'] if c != col])
        negative_list = df[df[f'diff_{col.lower()}'] <= 0].drop(
            columns=[f'diff_{c}' for c in ['d', 'b', 'cp', 'c', 'ooa', 'u'] if c != col])

        positive[col] = positive_list
        negative[col] = negative_list

    return positive, negative


def separate_smell_classes(df: pd.DataFrame) -> pd.DataFrame:
    # Separate the smell classes
    data = []

    for key, df in df.items():
        print(f"Positive & Negative {key}")
        print(f"shape: {df.shape[0]}")
        print(f"sum: {df[f'diff_{key}'].sum()}")

        for h in df['time_month_class'].unique():
            for year in df['year'].unique():
                class_df = df[(df['time_month_class'] == h) & (df['year'] == year)]
                data.append({
                    'smell_key': key,
                    'time_month_class': h,
                    'year': year,
                    'sum_diff': class_df[f'diff_{key}'].sum(),
                    'shape': class_df.shape[0]
                })

    return pd.DataFrame(data)


def separant_calculate_smell(df_positive: pd.DataFrame, df_negative: pd.DataFrame) -> pd.DataFrame:
    df_positive['percentage_positive'] = df_positive['sum_diff'] / df_positive['shape']
    df_negative['percentage_negative'] = df_negative['sum_diff'] / df_negative['shape']

    return pd.merge(df_positive, df_negative, on=['time_month_class', 'smell_key', 'year'], how='inner')


def plot_compare_smell_bug(df: pd.DataFrame, project_name: str):
    # Filter the data for each smell type
    smell_types = ['b', 'd', 'c', 'cp', 'ooa', 'u']
    smell_data = {smell: df[df['smell_key'] == smell] for smell in smell_types}

    # Plotting positive and negative values for each smell type over time month classes and years
    for smell, data in smell_data.items():
        plt.figure(figsize=(12, 6))

        # Group by time_month_class and year and sort by these columns
        grouped_data = data.groupby(['time_month_class', 'year']).sum().reset_index().sort_values(
            by=['year', 'time_month_class'])

        # Create a combined x-axis label
        x_labels = [f'{int(month)}-{int(year)}' for month, year in
                    zip(grouped_data['time_month_class'], grouped_data['year'])]

        plt.bar(x_labels, grouped_data['sum_diff_x'],
                color='green', label=f'Smell {smell.upper()} Positive',
                width=0.4,
                alpha=0.5
                )
        plt.bar(x_labels, grouped_data['sum_diff_y'],
                color='red', label=f'Smell {smell.upper()} Negative',
                width=0.4,
                alpha=0.5
                )

        plt.title(f'{project_name} - Smell {smell.upper()} Positive and Negative total smell have Bug each time month')

        plt.xlabel('Time Month Class-Year')
        plt.ylabel('Total Smell')
        plt.xticks(rotation=45)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        # plt.savefig(os.path.join(f'{project_name}_compare_smell_{smell}_bug.png'))
        plt.show()


def plot_diff_bug_over_time(df: pd.DataFrame, project_name: str):
    # Sort the DataFrame by completed_date in ascending order
    df = df.sort_values(by='completed_date')

    plt.figure(figsize=(12, 6))

    # Plot diff_bug over completed_date
    plt.plot(df['completed_date'], df['diff_bug'], marker='o', linestyle='-', color='b', label='diff_bug')

    plt.title(f'{project_name} - diff bug over Time')
    plt.xlabel('Completed Date')
    plt.ylabel('diff_bug')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(f'{project_name}_diff_bug_over_time.png'))
    plt.show()


if __name__ == '__main__':
    # Load the data
    sonar_smell_bug_seatunnel = pd.read_pickle('../../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    sonar_smell_bug_seatunnel_outliers = pd.read_pickle(
        '../../models/robust_outliers/output/seatunnal_bug_comapare_time_robuts_outlier.pkl')

    # Calculate the smell and bug values
    seatunnal = calculate_smell_bug(sonar_smell_bug_seatunnel)
    seatunnal_time = custom_time_hour_clustering_year(seatunnal)
    seatunnal_time = custom_time_clustering_media_average(seatunnal_time)
    seatunnal_positive, seatunnal_negative = separate_smell_integer(seatunnal_time)
    seatunnal_positive = separate_smell_classes(seatunnal_positive)
    seatunnal_negative = separate_smell_classes(seatunnal_negative)
    seatunnal_diff = separant_calculate_smell(seatunnal_positive, seatunnal_negative)

    seatunnal_outliers = calculate_smell_bug(sonar_smell_bug_seatunnel_outliers)
    seatunnal_time_outliers = custom_time_hour_clustering_year(seatunnal_outliers)
    seatunnal_time_outliers = custom_time_clustering_media_average(seatunnal_time_outliers)
    seatunnal_positive_outliers, seatunnal_negative_outliers = separate_smell_integer(seatunnal_time_outliers)
    seatunnal_positive_outliers = separate_smell_classes(seatunnal_positive_outliers)
    seatunnal_negative_outliers = separate_smell_classes(seatunnal_negative_outliers)
    seatunnal_diff_outliers = separant_calculate_smell(seatunnal_positive_outliers, seatunnal_negative_outliers)

    # plot_compare_smell_bug(seatunnal_diff, 'seatunnal')
    plot_diff_bug_over_time(seatunnal, 'seatunnal')
    plot_diff_bug_over_time(seatunnal_outliers, 'seatunnal_outliers')
