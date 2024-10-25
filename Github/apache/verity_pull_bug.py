import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spyder.utils.introspection.rope_patch import apply


def merge_dataframes(dfs_outlier: dict, dfs_pull_bug_issues: dict, repo: str) -> pd.DataFrame:
    return pd.merge(dfs_outlier[repo], dfs_pull_bug_issues[repo], on=['url', 'merge_commit_sha', 'base.sha'],
                    how='inner')


def filter_bugs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
        items=['url', 'id', 'issue_url', 'number', 'state', 'created_at', 'updated_at', 'closed_at', 'merged_at_x',
               'merged_at_y',
               'merge_commit_sha', 'labels', 'review_comments_url', 'review_comment_url', 'comments_url', 'merged',
               'comments', 'review_comments', 'commits', 'additions', 'deletions', 'changed_files', 'total_time',
               'total_time_hours', 'created_project', 'created_Bloaters', 'created_Change Preventers',
               'created_Couplers', 'created_Dispensables', 'created_Object-Orientation Abusers',
               'created_Uncategorized', 'ended_Bloaters', 'ended_Change Preventers', 'ended_Couplers',
               'ended_Dispensables', 'ended_Object-Orientation Abusers', 'ended_Uncategorized'])

    df['year'] = pd.to_datetime(df['merged_at_y']).dt.year
    return df


def custom_time_hour_clustering(df: pd.DataFrame) -> pd.DataFrame:
    # Define the bin edges and labels for months
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    labels = list(range(12))

    # Extract month from the 'merged_at_y' column
    df['month'] = pd.to_datetime(df['merged_at_y']).dt.month

    # Use pd.cut to assign values to the appropriate bins
    df['time_month_class'] = pd.cut(df['month'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def custom_time_hour_clustering_year(df: pd.DataFrame) -> pd.DataFrame:
    # Define the bin edges and labels for months
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    labels = list(range(12))

    # Extract month from the 'merged_at' column
    df['month'] = pd.to_datetime(df['merged_at']).dt.month
    df['year'] = pd.to_datetime(df['merged_at']).dt.year

    # Use pd.cut to assign values to the appropriate bins
    df['time_month_class'] = pd.cut(df['month'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def custom_time_clustering_media_average(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'merged_at' is converted to datetime
    df['merged_at'] = pd.to_datetime(df['merged_at'])

    # Calculate median and average time
    median_time = df['merged_at'].median()
    average_time = df['merged_at'].mean()

    # Assign median and average time to the DataFrame
    df['time_class_median'] = df['merged_at'].apply(lambda x: 0 if x > median_time else 1 if x < median_time else 2)
    df['time_class_average'] = df['merged_at'].apply(lambda x: 0 if x > average_time else 1 if x < average_time else 2)
    return df


def percentage_smell(df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the percentage change for each smell type
    rename_dict = {
        'created_Dispensables': 'created_d',
        'created_Bloaters': 'created_b',
        'created_Change Preventers': 'created_cp',
        'created_Couplers': 'created_c',
        'created_Object-Orientation Abusers': 'created_ooa',
        'created_Uncategorized': 'created_u',
        'ended_Dispensables': 'ended_d',
        'ended_Bloaters': 'ended_b',
        'ended_Change Preventers': 'ended_cp',
        'ended_Couplers': 'ended_c',
        'ended_Object-Orientation Abusers': 'ended_ooa',
        'ended_Uncategorized': 'ended_u'
    }
    df = df.rename(columns=rename_dict)

    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'diff_{col.lower()}'] = df[f'ended_{col}'] - df[f'created_{col}']
        df[f'percentage_{col.lower()}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

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


def separate_times(df: pd.DataFrame) -> pd.DataFrame:
    data = []
    for h in df['time_month_class'].unique():
        for year in df['year'].unique():
            class_df = df[
                (df['time_month_class'] == h) & (df['year'] == year)]
            data.append({
                'time_month_class': h,
                'year': year,
                'shape': class_df.shape[0]
            })

    return pd.DataFrame(data)


def separate_times_median(df: pd.DataFrame) -> pd.DataFrame:
    data = []
    for h in df['time_class_median'].unique():
        for year in df['year'].unique():
            class_df = df[
                (df['time_class_median'] == h) & (df['year'] == year)]
            data.append({
                'time_class_median': h,
                'year': year,
                'shape': class_df.shape[0]
            })

    return pd.DataFrame(data)


def separate_times_avg(df: pd.DataFrame) -> pd.DataFrame:
    data = []
    for h in df['time_class_average'].unique():
        for year in df['year'].unique():
            class_df = df[
                (df['time_class_average'] == h) & (df['year'] == year)]
            data.append({
                'time_class_average': h,
                'year': year,
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
        plt.savefig(os.path.join(f'{project_name}_compare_smell_{smell}.png'))
        plt.show()


def plot_compare_smell(df: pd.DataFrame, project_name: str):
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

        plt.title(f'{project_name} - Smell {smell.upper()} Positive and Negative total smell each time month')

        plt.xlabel('Time Month Class-Year')
        plt.ylabel('Total Smell')
        plt.xticks(rotation=45)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        plt.savefig(os.path.join(f'{project_name}_compare_smell_{smell}.png'))
        plt.show()


def plot_compare_time(df: pd.DataFrame, project_name: str):
    if 'time_month_class' not in df.columns:
        raise KeyError("The column 'time_month_class' does not exist in the DataFrame.")

    plt.figure(figsize=(12, 6))

    # Group by time_month_class and year and sort by these columns
    grouped_data = df.groupby(['time_month_class', 'year']).sum().reset_index().sort_values(
        by=['year', 'time_month_class'])

    # Create a combined x-axis label
    x_labels = [f'{int(month)}-{int(year)}' for month, year in
                zip(grouped_data['time_month_class'], grouped_data['year'])]

    plt.bar(x_labels, grouped_data['shape'],
            color='green', label='Shape',
            width=0.4,
            alpha=0.5)

    plt.title(f'{project_name} - Total Shape of Each Time Month')
    plt.xlabel('Time Month - Year')
    plt.ylabel('Total Shape')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(f'{project_name}_compare_shape.png'))
    plt.show()

# time_class_median

def plot_compare_time_class_median(df: pd.DataFrame, project_name: str):
    if 'time_class_median' not in df.columns:
        raise KeyError("The column 'time_class_average' does not exist in the DataFrame.")

    plt.figure(figsize=(12, 6))

    # Group by time_month_class and year and sort by these columns
    grouped_data = df.groupby(['time_class_median', 'year']).sum().reset_index().sort_values(
        by=['year', 'time_class_median'])

    # Create a combined x-axis label
    x_labels = [f'{int(month)}-{int(year)}' for month, year in
                zip(grouped_data['time_class_median'], grouped_data['year'])]

    plt.bar(x_labels, grouped_data['shape'],
            color='green', label='Shape',
            width=0.4,
            alpha=0.5)

    plt.title(f'{project_name} - Total Shape of Each Time Month')
    plt.xlabel('Time Month - Year')
    plt.ylabel('Total Shape')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(f'{project_name}_compare_shape.png'))
    plt.show()


def plot_compare_time_class_average(df: pd.DataFrame, project_name: str):
    if 'time_class_average' not in df.columns:
        raise KeyError("The column 'time_class_average' does not exist in the DataFrame.")

    plt.figure(figsize=(12, 6))

    # Group by time_month_class and year and sort by these columns
    grouped_data = df.groupby(['time_class_average', 'year']).sum().reset_index().sort_values(
        by=['year', 'time_class_average'])

    # Create a combined x-axis label
    x_labels = [f'{int(month)}-{int(year)}' for month, year in
                zip(grouped_data['time_class_average'], grouped_data['year'])]

    plt.bar(x_labels, grouped_data['shape'],
            color='green', label='Shape',
            width=0.4,
            alpha=0.5)

    plt.title(f'{project_name} - Total Shape of Each Time Month')
    plt.xlabel('Time Month - Year')
    plt.ylabel('Total Shape')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(f'{project_name}_compare_shape.png'))
    plt.show()


def data_collection_bug(df: pd.DataFrame, project_name: str) -> pd.DataFrame:
    positive, negative = separate_smell_integer(df)
    positive_class = separate_smell_classes(positive)
    negative_class = separate_smell_classes(negative)
    smell = separant_calculate_smell(positive_class, negative_class)
    plot_compare_smell(smell, project_name)
    return smell


if __name__ == '__main__':
    pull_commens_files = {
        'pulsar': '../output/pulsar_filtered_issues_requests_comments.pkl',
        'ozone': '../output/ozone_filtered_issues_requests_comments.pkl',
        'seatunnel': '../output/seatunnel_filtered_issues_requests_comments.pkl'
    }

    pull_bug_files = {
        'pulsar': '../output/pulsar_bug_pull_requests.pkl',
        'ozone': '../output/ozone_bug_pull_requests.pkl',
        'seatunnel': '../output/seatunnel_bug_pull_requests.pkl'
    }

    pull_bug_issues_files = {
        'pulsar': '../output/pulsar_filtered_issue_bug.pkl',
        'ozone': '../output/ozone_filtered_issue_bug.pkl',
        'seatunnel': '../output/seatunnel_filtered_issue_bug.pkl'
    }

    outlier = {
        'pulsar': '../../models/output/pulsar_prepare_to_train_newversion_9Sep.parquet',
        'ozone': '../../models/output/ozone_prepare_to_train_newversion_9Sep.parquet',
        'seatunnel': '../../models/output/seatunnel_prepare_to_train_newversion_9Sep.parquet'
    }

    dfs_pull_commens = {repo: pd.read_pickle(file) for repo, file in pull_commens_files.items()}
    dfs_pull_bug = {repo: pd.read_pickle(file) for repo, file in pull_bug_files.items()}
    dfs_pull_bug_issues = {repo: pd.read_pickle(file) for repo, file in pull_bug_issues_files.items()}
    dfs_outlier = {repo: pd.read_parquet(file) for repo, file in outlier.items()}

    pulsar_bug = merge_dataframes(dfs_outlier, dfs_pull_bug_issues, 'pulsar')
    ozone_bug = merge_dataframes(dfs_outlier, dfs_pull_bug_issues, 'ozone')
    seatunnel_bug = merge_dataframes(dfs_outlier, dfs_pull_bug_issues, 'seatunnel')

    pulsar_bugs = filter_bugs(pulsar_bug)
    ozone_bugs = filter_bugs(ozone_bug)
    seatunnel_bugs = filter_bugs(seatunnel_bug)

    pulsar = percentage_smell(custom_time_hour_clustering(pulsar_bugs))
    ozone = percentage_smell(custom_time_hour_clustering(ozone_bugs))
    seatunnel = percentage_smell(custom_time_hour_clustering(seatunnel_bugs))

    # pulsar_smell_bug = data_collection_bug(pulsar, 'Pulsar')
    # ozone_smell_bug = data_collection_bug(ozone, 'Ozone')
    # seatunnel_smell_bug = data_collection_bug(seatunnel, 'Seatunnel')

    pulsar_issue_bug = custom_time_hour_clustering_year(dfs_pull_bug_issues['pulsar'])
    pulsar_issue_bug_median_avg = custom_time_clustering_media_average(pulsar_issue_bug)
    pulsar_issue_bug_time = separate_times(pulsar_issue_bug)
    pulsar_issue_bug_time_media = separate_times_median(pulsar_issue_bug_median_avg)
    pulsar_issue_bug_time_avg = separate_times_avg(pulsar_issue_bug_median_avg)
    # plot_compare_time(pulsar_issue_bug_time, 'pulsar')
    # plot_compare_time_class_median(pulsar_issue_bug_time_media, 'pulsar-time-media')
    plot_compare_time_class_average(pulsar_issue_bug_time_avg, 'pulsar-time-average')

    #
    ozone_issue_bug = custom_time_hour_clustering_year(dfs_pull_bug_issues['ozone'])
    ozone_issue_bug_median_avg = custom_time_clustering_media_average(ozone_issue_bug)
    ozone_issue_bug_time = separate_times(ozone_issue_bug)
    ozone_issue_bug_time_median = separate_times_median(ozone_issue_bug_median_avg)
    ozone_issue_bug_time_avg = separate_times_avg(ozone_issue_bug_median_avg)
    # plot_compare_time(ozone_issue_bug_time, 'ozone')
    # plot_compare_time_class_median(ozone_issue_bug_time_median, 'ozone-time-media')
    plot_compare_time_class_average(ozone_issue_bug_time_avg,'ozone-time-average')

    seatunnel_issue_bug = custom_time_hour_clustering_year(dfs_pull_bug_issues['seatunnel'])
    seatunnel_issue_bug_median_avg = custom_time_clustering_media_average(seatunnel_issue_bug)
    seatunnel_issue_bug_time = separate_times(seatunnel_issue_bug)
    seatunnel_issue_bug_time_median = separate_times_median(seatunnel_issue_bug_median_avg)
    seatunnel_smell_bug_time_avg = separate_times_avg(seatunnel_issue_bug_median_avg)
    # plot_compare_time(seatunnel_issue_bug_time, 'seatunnel')
    # plot_compare_time_class_median(seatunnel_issue_bug_time_median, 'seatunnel-time-media')
    plot_compare_time_class_average(seatunnel_smell_bug_time_avg, 'seatunnel-time-average')

