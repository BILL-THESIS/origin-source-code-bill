import pandas as pd

def extract_start_pull_request(df_repository):
    created = df_repository[['url', 'base.sha']]
    ended = df_repository[['url', 'merge_commit_sha']]
    return created, ended


def extract_smells_pull_request(df_sha_pull, df_all_projects_sonar, sha_column):
    return pd.merge(df_sha_pull, df_all_projects_sonar, left_on=sha_column, right_on='revision')


def extract_smells_index(df_merged_sha, df_sonar_issues_smells):
    return pd.merge(df_merged_sha, df_sonar_issues_smells, on='key')


def prefix_columns(df, prefix):
    df.columns = [f"{prefix}_{col}" for col in df.columns]
    return df


def process_project(df_outliers, df_sonar, df_category_smells, project_name):
    created, ended = extract_start_pull_request(df_outliers)

    created_smells = extract_smells_pull_request(created, df_sonar, 'base.sha')
    ended_smells = extract_smells_pull_request(ended, df_sonar, 'merge_commit_sha')

    merged_smells_created = extract_smells_index(created_smells, df_category_smells)
    merged_smells_ended = extract_smells_index(ended_smells, df_category_smells)

    created_col = prefix_columns(merged_smells_created, "created")
    ended_col = prefix_columns(merged_smells_ended, "ended")


    merged_step1 = pd.merge(df_outliers, created_col, left_on=['url', 'base.sha'], right_on=['created_url', 'created_revision_x'], how='inner')
    merged_step2 = pd.merge(merged_step1, ended_col, left_on=['url', 'merge_commit_sha'], right_on=['ended_url', 'ended_revision_x'], how='inner')

    final_df = merged_step2.drop([
        'base.repo.created_at', 'base.repo.updated_at', 'base.repo.pushed_at',
        'created_url', 'created_base.sha',
        'created_project',
        'created_key',
        'created_name_x',
        'created_qualifier_x',
        'created_visibility_x', 'created_lastAnalysisDate_x',
        # 'created_revision_x',
        'created_name_y',
        'created_qualifier_y', 'created_visibility_y', 'created_lastAnalysisDate_y', 'created_revision_y',
        'ended_url', 'ended_merge_commit_sha',
        'ended_key',
        'ended_name_x', 'ended_qualifier_x',
        'ended_visibility_x', 'ended_lastAnalysisDate_x', 'ended_revision_x', 'ended_name_y',
        'ended_qualifier_y', 'ended_visibility_y', 'ended_lastAnalysisDate_y',
        # 'ended_revision_y',
        'ended_project'
    ], axis=1)

    # final_df.drop_duplicates(inplace=True)

    # final_df.to_parquet(f"../models/output/{project_name}_prepare_to_train_9sep.parquet")
    return final_df

if __name__ == "__main__":
    df_pulsar_outliers = pd.read_parquet("../models/output/pulsar_filtered_robust_outlier.parquet")
    df_seatunnel_outliers = pd.read_parquet("../models/output/seatunnel_filtered_robust_outlier_new.parquet")
    df_ozone_outliers = pd.read_parquet("../models/output/ozone_filtered_robust_outlier.parquet")
    # df_ozone_outliers_newversion = pd.read_parquet("../models/output/ozone_filtered_robust_outlier_newversion.parquet")

    df_sonar = pd.read_pickle("../Sonar/output/sonar_all_projects_version3.pkl")
    df_sonar_v2 = pd.read_pickle("../Sonar/output/sonar_all_projects_version4.pkl")
    df_sonar_v5 = pd.read_pickle("../Sonar/output/sonar_all_projects_version5.pkl")
    df_sonar_v6 = pd.read_pickle("../Sonar/output/sonar_all_projects_version6.pkl")


    df_rule_ = pd.read_pickle("../Sonar/output/sonar_rules_version9.9.6.pkl")

    df_sonar_issues = pd.read_pickle("../Sonar/output/sonar_smells_all_project.pkl")
    df_sonar_issues_v2 = pd.read_pickle("../Sonar/output/sonar_smells_all_project_version2.pkl")
    df_sonar_issues_v3 = pd.read_pickle("../Sonar/output/sonar_smells_all_project_version5.pkl")
    df_sonar_issues_v6 = pd.read_pickle("../Sonar/output/sonar_smells_all_project_version6.pkl")

    df_category_smells = pd.read_pickle("../Sonar/output/sonar_group_rules_category_smells.pkl")
    df_category_smells_v2 = pd.read_pickle("../Sonar/output/sonar_group_rules_category_smells_version2.pkl")
    df_category_smells_v5 = pd.read_pickle("../Sonar/output/sonar_group_rules_category_smells_version5.pkl")
    df_category_smells_v6 = pd.read_pickle("../Sonar/output/sonar_group_rules_category_smells_version6.pkl")

    sonar_name = pd.concat([df_sonar, df_sonar_v2, df_sonar_v5], axis=0)
    sonar_issues = pd.concat([df_sonar_issues, df_sonar_issues_v2,df_sonar_issues_v3], axis=0)
    sonar_category = pd.concat([df_category_smells, df_category_smells_v2, df_category_smells_v5], axis=0)


    ozone = process_project(df_ozone_outliers, sonar_name, sonar_category, "ozone")
    ozone_drop = ozone.drop_duplicates(ignore_index=True)
    ozone_all_duplicated = ozone_drop[ozone_drop.duplicated(subset=['url','base.sha', 'merge_commit_sha'], keep=False)]
    ozone_m1 = ozone_drop.drop_duplicates(subset=['url','base.sha', 'merge_commit_sha'], keep='first')
    # verify if there are duplicates
    ozone_m2 = ozone_m1[ozone_m1['merge_commit_sha'] == ozone_m1['ended_revision_y']]
    ozone_m3 = ozone_m1[ozone_m1['base.sha'] == ozone_m1['created_revision_x']]

    pulsar = process_project(df_pulsar_outliers, sonar_name, sonar_category, "pulsar")
    pulasr_drop = pulsar.drop_duplicates()
    pulsar_all_duplicated = pulasr_drop[pulasr_drop.duplicated(subset=['url','base.sha', 'merge_commit_sha'], keep=False)]
    pulsar_m1 = pulasr_drop.drop_duplicates(subset=['url','base.sha', 'merge_commit_sha'], keep='first')

    seatunnel = process_project(df_seatunnel_outliers, sonar_name, sonar_category, "seatunnel")
    seatunnel_drop = seatunnel.drop_duplicates()
    seatunnel_all_duplicated = seatunnel_drop[seatunnel_drop.duplicated(subset=['url','base.sha', 'merge_commit_sha'], keep=False)]
    seatunnel_m1 = seatunnel_drop.drop_duplicates(subset=['url','base.sha', 'merge_commit_sha'], keep='first')
