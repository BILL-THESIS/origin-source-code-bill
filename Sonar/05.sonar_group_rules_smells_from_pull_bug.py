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
    df_sonar_rules_cat_smells = pd.read_parquet("../Sonar/output/sonar_rules_categorized.parquet")
    df_sonar_search_pull = pd.read_pickle("../Sonar/output/01.sonar_search_pull_bug.pkl")
    df_sonar_smell = pd.read_pickle("../Sonar/output/02.sonar_bug_pull_smell.pkl")
    df_sonar_smell_reindex = df_sonar_smell.reset_index()

    df_sonar_smell_verify_key = pd.merge(df_sonar_search_pull, df_sonar_smell, left_on='key', right_index=True)

    # Pivot the data frame to have counts for each smell category as columns
    category_rules = df_sonar_rules_cat_smells.pivot(index='key', columns='category', values='key').fillna(0).reset_index()

    # divided the data frame into groups
    grouped_smells = create_grouped_dfs(category_rules)
    grouped_smells_list = transform_data(grouped_smells)

    melt = pd.melt(category_rules, id_vars='key', var_name='category', value_name='count')
    melt = melt[melt['count'] != 0]
    melt = melt.reset_index(drop=True)

    # Melt the Left DataFrame to long format
    df_rules_melted = (df_sonar_smell_reindex.melt(id_vars=['project'], var_name='rule', value_name='count'))
    merged_df = df_rules_melted.merge(melt, how='left', left_on='rule', right_on='key')

    # Pivoting the merged DataFrame to get the desired format
    result_df = merged_df.pivot_table(index='project', columns='category', values='count_x', aggfunc='sum').reset_index()
    result_final = pd.merge(df_sonar_search_pull, result_df, left_on='key', right_on='project')

    result_final.to_pickle("../Sonar/output/05.sonar_group_rules_category_smells_pull_bug.pkl")

