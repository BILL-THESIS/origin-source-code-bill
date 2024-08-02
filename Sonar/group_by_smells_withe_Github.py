import pandas as pd


def extract_start_pull_request(df_repository):
    created = df_repository[['url', 'base.sha']]
    ended = df_repository[['url', 'merge_commit_sha']]
    return created, ended


def extract_smells_pull_request_created(df_sha_pull, df_all_projects_sonar):
    merged_df_created = pd.merge(df_sha_pull, df_all_projects_sonar, left_on='base.sha', right_on='revision')
    return merged_df_created


def extract_smells_pull_request_ended(df_sha_pull, df_all_projects_sonar):
    merged_df_ended = pd.merge(df_sha_pull, df_all_projects_sonar, left_on='merge_commit_sha', right_on='revision')
    return merged_df_ended


def extract_smells_index(df_merged_sha, df_sonar_issues_smells):
    merged_df = pd.merge(df_merged_sha, df_sonar_issues_smells, left_on='key', right_index=True)
    return merged_df


def prefix_columns_created(df):
    df.columns = [f"created_{col}" for col in df.columns]
    return df


def prefix_columns_ended(df):
    df.columns = [f"ended_{col}" for col in df.columns]
    return df


if __name__ == "__main__":
    df_pulsar_outliers = pd.read_parquet("../models/KMeans/output/pulsar_filtered_robust_outlier.parquet")
    df_seatunnel_outliers = pd.read_parquet("../models/KMeans/output/seatunnel_filtered_robust_outlier.parquet")
    df_ozone_outliers = pd.read_parquet("../models/KMeans/output/ozone_filtered_robust_outlier.parquet")

    df_sonar = pd.read_pickle("../Sonar/output/sonar_all_projects_version3.pkl")
    df_rule_ = pd.read_pickle("../Sonar/output/sonar_rules_version9.9.6.pkl")
    df_sonar_issues = pd.read_pickle("../Sonar/output/sonar_smells_all_project.pkl")

    ozone_created, ozone_ended = extract_start_pull_request(df_ozone_outliers)
    ozone_created_smells = extract_smells_pull_request_created(ozone_created, df_sonar)
    ozone_ended_smells = extract_smells_pull_request_ended(ozone_ended, df_sonar)
    ozone_merged_smells_created = extract_smells_index(ozone_created_smells, df_sonar_issues)
    ozone_merged_smells_ended = extract_smells_index(ozone_ended_smells, df_sonar_issues)
