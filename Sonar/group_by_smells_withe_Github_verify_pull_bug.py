import pandas as pd


def merge_seatunnal_data(df_pull, df_sonar_bug_pull):
    strat = pd.merge(df_pull, df_sonar_bug_pull, left_on='base.sha', right_on='revision', how='left')
    end = pd.merge(df_pull, df_sonar_bug_pull, left_on='merge_commit_sha', right_on='revision', how='left')
    return strat, end


def select_columns(df_start, df_end):
    start = df_start[['url', 'base.sha', 'created_at',
                      'merge_commit_sha', 'merged_at',
                      'comments', 'review_comments',
                      'commits',
                      'additions', 'deletions', 'changed_files',
                      'key', 'revision', 'metric',
                      'value']]
    end = df_end[['url', 'base.sha', 'created_at',
                  'merge_commit_sha', 'merged_at',
                  'comments', 'review_comments',
                  'commits',
                  'additions', 'deletions', 'changed_files',
                  'key', 'revision', 'metric',
                  'value']]
    return start, end


def merge_selected_columns(df_start, df_end):
    df_compare_bugs = pd.merge(df_start, df_end, on=['url', 'base.sha', 'created_at',
                                                     'merge_commit_sha', 'merged_at',
                                                     'comments', 'review_comments',
                                                     'commits',
                                                     'additions', 'deletions', 'changed_files'],
                               suffixes=('_created', '_ended'))
    return df_compare_bugs

def merge_with_category_smell(df, data_category_smell):
    return pd.merge(df, data_category_smell, on=['key', 'revision'], how='inner')

if __name__ == "__main__":
    # data pull requests have conversations and tag bugs issues
    df_pulsar = pd.read_pickle("../Github/output/pulsar_filtered_issue_bug.pkl")
    df_seatunnel = pd.read_pickle("../Github/output/seatunnel_filtered_issue_bug.pkl")
    df_ozone = pd.read_pickle("../Github/output/ozone_filtered_issue_bug.pkl")

    # Extractions code smell and number of bug via SonarQube
    df_sonar = pd.read_pickle("../Sonar/output/01.sonar_search_pull.pkl")
    df_sonar_bug_measures = pd.read_pickle('../Sonar/output/02.sonar_measures_bugs.pkl')
    df_sonar_smell_measures = pd.read_pickle('../Sonar/output/02.sonar_measures_smells.pkl')
    data_category_smell = pd.read_pickle('../Sonar/output/05.sonar_group_rules_category_smells_pull_bug.pkl')

    df_sonar_bug_pull = pd.merge(df_sonar, df_sonar_bug_measures, left_on='key', right_on='component.key')

    # Transform the data frame to have the smell categories as columns
    seatunnal_strat_bug, seatunnal_end_bug = merge_seatunnal_data(df_seatunnel, df_sonar_bug_pull)
    seatunnal_strat, seatunnal_end = select_columns(seatunnal_strat_bug, seatunnal_end_bug)
    seatunnal_strat_pull = merge_with_category_smell(seatunnal_strat, data_category_smell)
    seatunnal_end_pull = merge_with_category_smell(seatunnal_end, data_category_smell)
    seatunnal_compare_bugs = merge_selected_columns(seatunnal_strat_pull, seatunnal_end_pull)
    seatunnal_compare_bugs.to_pickle('../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')

    ozone_start_bug, ozone_end_bug = merge_seatunnal_data(df_ozone, df_sonar_bug_pull)
    ozone_start, ozone_end = select_columns(ozone_start_bug, ozone_end_bug)
    ozone_start_pull = merge_with_category_smell(ozone_start, data_category_smell)
    ozone_end_pull = merge_with_category_smell(ozone_end, data_category_smell)
    ozone_compare_bugs = merge_selected_columns(ozone_start_pull, ozone_end_pull)
    ozone_compare_bugs.to_pickle('../Sonar/output/tag_bug/ozone_bug_comapare_time.pkl')
