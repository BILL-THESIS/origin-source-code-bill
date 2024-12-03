import pandas as pd


def verity_sha(df_pull, df_sonar_bug_pull):
    start_merge = pd.merge(df_pull, df_sonar_bug_pull, left_on='base.sha', right_on='revision', how='left')
    end_merge = pd.merge(df_pull, df_sonar_bug_pull, left_on='merge_commit_sha', right_on='revision', how='left')
    return start_merge, end_merge


def select_columns(df_start, df_end):
    columns = [
        'url', 'base.sha', 'created_at', 'merge_commit_sha', 'merged_at',
        'comments', 'review_comments', 'commits', 'additions', 'deletions',
        'changed_files', 'key', 'revision', 'smell_bug', 'smell_normal', 'smell_vulnerability'
    ]
    start_selected = df_start[columns]
    end_selected = df_end[columns]
    return start_selected, end_selected


def merge_selected_columns(df_start, df_end):
    merge_columns = [
        'url', 'base.sha', 'created_at', 'merge_commit_sha', 'merged_at',
        'comments', 'review_comments', 'commits', 'additions', 'deletions', 'changed_files'
    ]
    df_compare_bugs = pd.merge(df_start, df_end, on=merge_columns, suffixes=('_created', '_ended'), how='inner')
    return df_compare_bugs


def calculate_time_smell(df: pd.DataFrame) -> pd.DataFrame:
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
    df['completed_date'] = pd.to_datetime(df['created_at']) + df['total_time']
    df['diff_normal'] = df['smell_normal_ended'] - df['smell_normal_created']
    df['diff_bug'] = df['smell_bug_ended'] - df['smell_bug_created']
    df['diff_vulnerability'] = df['smell_vulnerability_ended'] - df['smell_vulnerability_created']
    return df

def verify_project_keys(data_compare : pd.DataFrame, project_name: str) -> pd.DataFrame:
    key_create = data_compare[~data_compare['key_created'].isin(data_compare['base.sha'])]
    key_end = data_compare[~data_compare['key_ended'].isin(data_compare['merge_commit_sha'])]
    key_have_data = data_compare[
        data_compare.apply(lambda row: row['key_created'] in data_compare['base.sha'].values and
                                       row['key_ended'] in data_compare['merge_commit_sha'].values, axis=1)
    ]
    key_create['base.sha'].to_csv(f"../beach_command/output/key_create_{project_name}.text", index=False, header=False)
    key_end['merge_commit_sha'].to_csv(f"../beach_command/output/key_end_{project_name}.text", index=False, header=False)
    return key_create, key_end, key_have_data

def verify_smell_vulnerability(data_compare : pd.DataFrame) -> pd.DataFrame:
    smell_vulnerability_created_nall = data_compare[data_compare['smell_vulnerability_created'].isna()]
    smell_vulnerability_ended_nall = data_compare[data_compare['smell_vulnerability_ended'].isna()]
    return smell_vulnerability_created_nall, smell_vulnerability_ended_nall


def save_smell_vulnerability(data_compare, project_name):
    print(str(f'{project_name} data:'))
    smell_vulnerability_created = data_compare[data_compare['smell_vulnerability_created'].isna()]
    # smell_vulnerability_created['base.sha'].to_csv(f"../beach_command/output/smell_vulnerability_created_{project_name}.text",
    #                                                index=False)
    smell_vulnerability_ended = data_compare[data_compare['smell_vulnerability_ended'].isna()]
    # smell_vulnerability_ended['merge_commit_sha'].to_csv(f"../beach_command/output/smell_vulnerability_ended_{project_name}.text",
    #                                                      index=False)

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

    # filter columns smell for each types for each project
    sonar_issues_smell_bug['smell_bug'] = sonar_issues_smell_bug.sum(axis=1)
    sonar_issues_smell_normal['smell_normal'] = sonar_issues_smell_normal.sum(axis=1)
    sonar_issues_smell_vulnerability['smell_vulnerability'] = sonar_issues_smell_vulnerability.sum(axis=1)

    # Remove duplicates
    drop_sonar_issues_name = sonar_issues_name.drop_duplicates(subset=['key'])
    drop_sonar_issues_smell_bug = sonar_issues_smell_bug.drop_duplicates(subset=['project'])
    drop_sonar_issues_smell_vulnerability = sonar_issues_smell_vulnerability.drop_duplicates(subset=['project'])
    drop_sonar_issues_smell_normal = sonar_issues_smell_normal.drop_duplicates(subset=['project'])

    # Merge filtered data
    sonar_issues_name_drop_filter = pd.merge(drop_sonar_issues_name, drop_sonar_issues_smell_normal,
                                             left_on='key', right_on='project', how='inner')
    sonar_issues_name_drop_filter = pd.merge(sonar_issues_name_drop_filter, drop_sonar_issues_smell_bug,
                                             on='project', how='left')
    sonar_issues_name_drop_filter = pd.merge(sonar_issues_name_drop_filter, drop_sonar_issues_smell_vulnerability,
                                             on='project', how='left')

    filter_col = sonar_issues_name_drop_filter[['key', 'revision', 'smell_bug', 'smell_normal', 'smell_vulnerability']]

    # Verify sha
    seatunnel_start, seatunnel_end = verity_sha(seatunnel, filter_col)
    seatunnel_start, seatunnel_end = select_columns(seatunnel_start, seatunnel_end)
    seatunnel_start['smell_vulnerability'].fillna(0, inplace=True)
    seatunnel_end['smell_vulnerability'].fillna(0, inplace=True)
    seatunnel_compare = calculate_time_smell(merge_selected_columns(seatunnel_start, seatunnel_end))
    seatunnel_compare.to_pickle('../models/output/data_prepare_models/seatunnel_compare_types_smells.pkl')


    ozone_start, ozone_end = verity_sha(ozone, filter_col)
    ozone_start, ozone_end = select_columns(ozone_start, ozone_end)
    ozone_compare = calculate_time_smell(merge_selected_columns(ozone_start, ozone_end))

    pulsar_start, pulsar_end = verity_sha(pulsar, filter_col)
    pulsar_start, pulsar_end = select_columns(pulsar_start, pulsar_end)
    pulsar_start['smell_vulnerability'].fillna(0, inplace=True)
    pulsar_end['smell_vulnerability'].fillna(0, inplace=True)
    pulsar_compare = calculate_time_smell(merge_selected_columns(pulsar_start, pulsar_end))

    # Verify data inside each project
    key_create_pulsar, key_end_pulsar, key_have_data_pulsar = verify_project_keys(pulsar_compare, 'pulsar')
    key_have_data_pulsar.to_pickle('../models/output/data_prepare_models/pulsar_compare_types_smells.pkl')


    smell_vulnerability_created_nall_seatunnel, smell_vulnerability_ended_nall_seatunnel = verify_smell_vulnerability(seatunnel_compare)

    # Verify data inside The columns Smell_vulnerability_created and Smell_vulnerability_ended
    save_smell_vulnerability(pulsar_compare, 'pulsar')

