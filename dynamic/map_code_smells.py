import pandas as pd


def factor_columns_github(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'url', 'base.sha', 'created_at', 'merge_commit_sha', 'merged_at',
        'comments', 'review_comments', 'commits', 'additions', 'deletions',
        'changed_files'
    ]

    return df[columns]


def verity_sha(df_pull, df_sonar_bug_pull):
    start_merge = pd.merge(df_pull, df_sonar_bug_pull, left_on='base.sha', right_on='revision', how='left')
    end_merge = pd.merge(df_pull, df_sonar_bug_pull, left_on='merge_commit_sha', right_on='revision', how='left')
    return start_merge, end_merge


def merge_selected_columns(df_start, df_end):
    merge_columns = [
        'url', 'base.sha', 'created_at', 'merge_commit_sha', 'merged_at',
        'comments', 'review_comments', 'commits', 'additions', 'deletions', 'changed_files'
    ]
    df_compare_bugs = pd.merge(df_start, df_end, on=merge_columns, suffixes=('_created', '_ended'), how='inner')

    for col in df_start.columns:
        if col in df_end.columns and pd.api.types.is_numeric_dtype(df_start[col]) and pd.api.types.is_numeric_dtype(
                df_end[col]):
            df_compare_bugs[f'{col}_diff'] = df_end[col] - df_start[col]

    return df_compare_bugs


def calculate_time_smell(df: pd.DataFrame) -> pd.DataFrame:
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
    df['completed_date'] = pd.to_datetime(df['created_at']) + df['total_time']
    return df


def verify_project_keys(data_compare: pd.DataFrame, project_name: str) -> pd.DataFrame:
    key_create = data_compare[~data_compare['key_created'].isin(data_compare['base.sha'])]
    key_end = data_compare[~data_compare['key_ended'].isin(data_compare['merge_commit_sha'])]
    key_have_data = data_compare[
        data_compare.apply(lambda row: row['key_created'] in data_compare['base.sha'].values and
                                       row['key_ended'] in data_compare['merge_commit_sha'].values, axis=1)
    ]
    return key_create, key_end, key_have_data


def verify_smell_vulnerability(data_compare: pd.DataFrame) -> pd.DataFrame:
    smell_vulnerability_created_nall = data_compare[data_compare['smell_vulnerability_created'].isna()]
    smell_vulnerability_ended_nall = data_compare[data_compare['smell_vulnerability_ended'].isna()]
    return smell_vulnerability_created_nall, smell_vulnerability_ended_nall


def save_smell_vulnerability(data_compare, project_name):
    print(str(f'{project_name} data:'))
    smell_vulnerability_created = data_compare[data_compare['smell_vulnerability_created'].isna()]
    smell_vulnerability_ended = data_compare[data_compare['smell_vulnerability_ended'].isna()]


if __name__ == "__main__":
    # Load the SonarQube issues data
    sonar_issues_name = pd.concat([
        pd.read_pickle("../Sonar/output/01.sonar_search_pull.pkl"),
        pd.read_pickle("../Sonar/output/01.sonar_search_pull_v2.pkl")
    ], axis=0)

    sonar_issues_smell_normal = pd.concat([
        pd.read_pickle("../Sonar/output/02.sonar_bug_pull_smell_v2.pkl"),
        pd.read_pickle("../Sonar/output/02.sonar_bug_pull_smell.pkl")
    ], axis=0).reset_index()

    sonar_issues_smell_bug = pd.concat([
        pd.read_pickle("../Sonar/output/02.sonar_issues_bugs.pkl"),
        pd.read_pickle("../Sonar/output/02.sonar_issues_bugs_v2.pkl")
    ], axis=0).reset_index()

    vulnerability_1 = pd.read_pickle("../Sonar/output/02.sonar_vulnerability_pull_smell.pkl").reset_index()
    vulnerability_2 = pd.read_pickle("../Sonar/output/02.sonar_vulnerability_pull_smell_v2.pkl").reset_index()
    vulnerability_3 = pd.read_pickle("../Sonar/output/02.sonar_vulnerability_pull_smell_v4.pkl").reset_index()

    # Remove the prefixes using a regular expression
    sonar_issues_name['key'] = sonar_issues_name['key'].str.replace(r'^(pulsar-|ozone-|seatunnel-)', '', regex=True)
    sonar_issues_smell_normal['project'] = sonar_issues_smell_normal['project'].str.replace(
        r'^(pulsar-|ozone-|seatunnel-)', '', regex=True)
    sonar_issues_smell_bug['project'] = sonar_issues_smell_bug['project'].str.replace(
        r'^(pulsar-|ozone-|seatunnel-)', '', regex=True)

    # Remove "pulsar-" prefix from the column 'project'
    vulnerability_2['project'] = vulnerability_2['project'].str.replace('pulsar-', '', regex=False)
    vulnerability_3['project'] = vulnerability_3['project'].str.replace(r'^(pulsar-|ozone-|seatunnel-)', '', regex=True)
    sonar_issues_smell_vulnerability = pd.concat([vulnerability_1, vulnerability_2, vulnerability_3], axis=0)

    print(sonar_issues_smell_normal['project'].duplicated().sum())
    print(sonar_issues_smell_bug['project'].duplicated().sum())
    print(sonar_issues_smell_vulnerability['project'].duplicated().sum())

    # Load Projects with Tags bugs in Pull Request
    pulsar = pd.read_pickle("../Github/output/pulsar_filtered_issue_bug.pkl")
    seatunnel = pd.read_pickle('../Github/output/seatunnel_filtered_issue_bug.pkl')
    ozone = pd.read_pickle('../Github/output/ozone_filtered_issue_bug.pkl')

    # Remove duplicates
    drop_sonar_issues_name = sonar_issues_name.drop_duplicates(subset=['key'])
    drop_sonar_issues_smell_bug = sonar_issues_smell_bug.drop_duplicates(subset=['project'])
    drop_sonar_issues_smell_vulnerability = sonar_issues_smell_vulnerability.drop_duplicates(subset=['project'])
    drop_sonar_issues_smell_normal = sonar_issues_smell_normal.drop_duplicates(subset=['project'])

    # Merge filtered data
    sonar_issues_drop_filter = pd.merge(drop_sonar_issues_name, drop_sonar_issues_smell_normal,
                                        left_on='key', right_on='project', how='inner')
    sonar_issues_drop_filter = pd.merge(sonar_issues_drop_filter, drop_sonar_issues_smell_bug,
                                        on='project', how='left')
    sonar_issues_drop_filter = pd.merge(sonar_issues_drop_filter, drop_sonar_issues_smell_vulnerability,
                                        on='project', how='left')

    # fillna in smell columns is 0
    sonar_issues_drop_filter_fill = sonar_issues_drop_filter.fillna(int(0))

    # Verify project keys
    seatunnel_query = factor_columns_github(seatunnel)
    seatunnel_start, seatunnel_end = verity_sha(seatunnel_query, sonar_issues_drop_filter_fill)
    seatunnel_compare = merge_selected_columns(seatunnel_start, seatunnel_end)
    seatunnel_compare = calculate_time_smell(seatunnel_compare)

    df = pd.DataFrame(seatunnel_compare.describe())


    pulsar_query = factor_columns_github(pulsar)
    pulsar_start, pulsar_end = verity_sha(pulsar_query, sonar_issues_drop_filter_fill)
    pulsar_compare = merge_selected_columns(pulsar_start, pulsar_end)
    pulsar_compare = calculate_time_smell(pulsar_compare)

    df_2 = pd.DataFrame(pulsar_compare.describe())
