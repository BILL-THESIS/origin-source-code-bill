import os
import numpy as np
import pandas as pd


def filter_pull(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
        items=['url', 'created_at', 'base.sha', 'merged_at', 'merge_commit_sha', 'merged',
               'comments', 'review_comments', 'commits', 'additions', 'deletions',
               'changed_files', ])

    df['year'] = pd.to_datetime(df['merged_at']).dt.year
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
        dfs_pull_bug_issues[repo]['base.sha'].drop_duplicates().to_csv(f'../../beach_command/output/{repo}_base_sha_drop.text', index=False)
        dfs_pull_bug_issues[repo]['merge_commit_sha'].drop_duplicates().to_csv(f'../../beach_command/output/{repo}_merge_commit_sha_drop.text', index=False)
        dfs_pull_bug_issues[repo]['base.sha'].to_csv(f'../../beach_command/output/{repo}_base_sha.text', index=False)
        dfs_pull_bug_issues[repo]['merge_commit_sha'].to_csv(f'../../beach_command/output/{repo}_merge_commit_sha.text', index=False)
