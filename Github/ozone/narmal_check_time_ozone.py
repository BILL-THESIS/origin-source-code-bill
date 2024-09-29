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
                               'review_comments',
                               'commits', 'additions', 'deletions', 'changed_files'])
    return df


if __name__ == '__main__':
    df = pd.read_pickle("../output/ozone_filtered_issues_requests_comments_pulls.pkl")

    re_df = query_data(df)

    re_df['total_time'] = pd.to_datetime(re_df['merged_at']) - pd.to_datetime(
        re_df['created_at'])

    # Converting total_time to hours for better readability
    re_df['total_time_hours'] = re_df['total_time'].dt.total_seconds() / 3600

    re_df.drop_duplicates()

    # re_df.to_pickle("../output/ozone_filtered_final_api_new.pkl")

    # Plotting total_time_hours
    plt.figure(figsize=(10, 6))
    sns.histplot(re_df['total_time_hours'], bins=50, kde=True)
    plt.xlabel('Total Time (hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Time from Creation to Merge in Ozone Repository')
    plt.show()
