import pandas as pd


def create_grouped_dfs(df):
    grouped_dfs = {}
    for col in df.columns[1:]:  # Skip the 'category' column
        unique_values = set(df[col]) - {'0'}
        grouped_dfs[col] = pd.DataFrame({col: list(unique_values)})
    return grouped_dfs


def transform_data(dict_group_smell):
    # Transform the data frame to have the smell categories as columns
    df = []
    for key, value in dict_group_smell.items():
        df.append(value)
    return df


if __name__ == "__main__":
    df_smells = pd.read_parquet("../Sonar/output/sonar_rules_categorized.parquet")
    df_sonar = pd.read_pickle("../Sonar/output/sonar_all_projects_version5.pkl")
    df_rules = pd.read_pickle("../Sonar/output/sonar_smells_all_project_version6.pkl")
    df_rules_reindex = df_rules.reset_index()

    df_sonar_smells = pd.merge(df_sonar, df_rules, left_on='key', right_index=True)
    # Pivot the data frame to have counts for each smell category as columns
    category_rules = df_smells.pivot(index='key', columns='category', values='key').fillna(0).reset_index()
    # divided the data frame into groups
    grouped_dfs = create_grouped_dfs(category_rules)
    grouped_list = transform_data(grouped_dfs)

    melt = pd.melt(category_rules, id_vars='key', var_name='category', value_name='count')
    melt = melt[melt['count'] != 0]
    melt = melt.reset_index(drop=True)

    # Melt the Left DataFrame to long format
    df_rules_melted = (df_rules_reindex.melt(id_vars=['project'], var_name='rule', value_name='count'))
    merged_df = df_rules_melted.merge(melt, how='left', left_on='rule', right_on='key')

    # Pivoting the merged DataFrame to get the desired format
    result_df = merged_df.pivot_table(index='project', columns='category', values='count_x', aggfunc='sum').reset_index()
    result_final = pd.merge(df_sonar, result_df, left_on='key', right_on='project')

    result_final.to_pickle("../Sonar/output/sonar_group_rules_category_smells_version6.pkl")

