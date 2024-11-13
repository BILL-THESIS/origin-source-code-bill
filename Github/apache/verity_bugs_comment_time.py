import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def filter_pull(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
        items=['url', 'created_at', 'base.sha', 'merged_at', 'merge_commit_sha', 'merged',
               'comments', 'review_comments', 'commits', 'additions', 'deletions',
               'changed_files']
    )

    df['year'] = pd.to_datetime(df['merged_at']).dt.year
    df['time_total'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])

    df['time_total_median'] = df['time_total'].median().days
    df['time_total_mean'] = df['time_total'].mean().days

    return df


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


    for repo in ['pulsar', 'ozone', 'seatunnel']:

        dfs_pull_bug_issues[repo] = filter_pull(dfs_pull_bug_issues[repo])
        df = dfs_pull_bug_issues[repo]

