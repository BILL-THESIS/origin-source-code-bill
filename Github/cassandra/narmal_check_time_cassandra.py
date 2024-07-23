import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def query_data(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(df,
                      columns=['url', 'merge_commit_sha', 'merged_at',
                               'created_at', 'base.sha',
                               'base.repo.created_at', 'base.repo.updated_at',
                               'base.repo.pushed_at',
                               'comments',
                               'review_comments', 'commits', 'additions', 'deletions', 'changed_files'])
    return df


if __name__ == '__main__':
    df = pd.read_pickle("../output/cassandra_filtered_issue_requests_comments_pulls.pkl")
    re_df = query_data(df)
    # Drop NaN values specifically in the 'merged_at' column
    re_df_drop_nan = re_df.dropna(subset=['merged_at'])

    re_df_drop_nan['total_time'] = pd.to_datetime(re_df_drop_nan['merged_at']) - pd.to_datetime(
        re_df_drop_nan['created_at'])
    # Converting total_time to hours for better readability
    re_df_drop_nan['total_time_hours'] = re_df_drop_nan['total_time'].dt.total_seconds() / 3600
    re_df_drop_nan.to_pickle("../output/cassandra_filtered_final_api.pkl")
    # Plotting total_time_hours
    plt.figure(figsize=(10, 6))
    sns.histplot(re_df_drop_nan['total_time_hours'], bins=50, kde=True)
    plt.xlabel('Total Time (hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Time from Creation to Merge on cassandra Repository')
    plt.show()
