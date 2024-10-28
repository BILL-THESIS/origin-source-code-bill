import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def custom_time_hour_clustering(df: pd.DataFrame) -> pd.DataFrame:
    # Define the bin edges and labels
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 12, 18, 24, 48, 72, 96, 120, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Use pd.cut to assign values to the appropriate bins
    df['time_hour_class'] = pd.cut(df['total_time_hours'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def load_and_prepare_data(file_paths: dict) -> dict:
    dfs = {repo: pd.read_pickle(file).filter(items=[
        'url', 'id', 'issue_url', 'number', 'state', 'created_at', 'updated_at', 'closed_at', 'merged_at',
        'merge_commit_sha', 'labels', 'review_comments_url', 'review_comment_url', 'comments_url', 'merged',
        'comments', 'review_comments', 'commits', 'additions', 'deletions', 'changed_files'])
        for repo, file in file_paths.items()}

    for repo, df in dfs.items():
        df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
        df['total_time_hours'] = df['total_time'].dt.total_seconds() / 3600
        df['year'] = pd.to_datetime(df['created_at']).dt.year

    return dfs


def separant_calculate_total_time(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'time_hour_class' and 'year', then calculate the sum of 'total_time_hours'
    grouped_df = df.groupby(['time_hour_class', 'year']).agg({'total_time_hours': 'sum'}).reset_index()
    return grouped_df


def plot_total_time(df: pd.DataFrame, repo: str):
    # Plotting the line graph
    plt.figure(figsize=(12, 8))
    for year in df['year'].unique():
        yearly_data = df[df['year'] == year]
        plt.plot(yearly_data['time_hour_class'], yearly_data['total_time_hours'], marker='o', label=f'Year {year}')

    plt.title(f' {repo} - Years by Time Hour Class')
    plt.xlabel('Time Hour Class')
    plt.ylabel('Total Time Hours')
    plt.xticks(ticks=range(15),
               labels=['< 0', '1', '2', '3', '4', '5', '6', '7-12', '12-18', '18-24', '24-48', '48-72', '72-96',
                       '96-120', '> 120'], rotation=45)
    plt.legend()
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':

    pull_bug_issues_files = {
        'pulsar': '../output/pulsar_filtered_issue_bug.pkl',
        'ozone': '../output/ozone_filtered_issue_bug.pkl',
        'seatunnel': '../output/seatunnel_filtered_issue_bug.pkl'
    }

    dfs_pull_bug_issues = load_and_prepare_data(pull_bug_issues_files)

    pulsar_bug = dfs_pull_bug_issues['pulsar']
    ozone_bug = dfs_pull_bug_issues['ozone']
    seatunnel_bug = dfs_pull_bug_issues['seatunnel']

    # Apply the function to the DataFrame
    p_total_time = separant_calculate_total_time(custom_time_hour_clustering(pulsar_bug))
    o_total_time = separant_calculate_total_time(custom_time_hour_clustering(ozone_bug))
    s_total_time = separant_calculate_total_time(custom_time_hour_clustering(seatunnel_bug))

    plot_total_time(p_total_time, 'Pulsar')
    plot_total_time(o_total_time, 'Ozone')
    plot_total_time(s_total_time, 'Seatunnel')

