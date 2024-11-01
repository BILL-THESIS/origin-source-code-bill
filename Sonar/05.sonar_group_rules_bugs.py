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
    df_smell = pd.read_pickle("../Sonar/output/sonar_bug_pull_smell.pkl")
    df_pull = pd.read_pickle("../Sonar/output/sonar_all_projects_bug_pull.pkl")
    df_bug = pd.read_pickle("../Sonar/output/sonar_bug.pkl")

