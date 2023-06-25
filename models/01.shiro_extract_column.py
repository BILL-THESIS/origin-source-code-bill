import pandas as pd
from datetime import timedelta

git_df = pd.read_csv("../Github/output/03.shiro_label.csv")
# git_df1 = pd.read_csv("../Github/output/05.shiro.csv")
# git_df2 = pd.read_csv("../Github/output/03.url_to_pull_353.csv")
# git_df3 = pd.read_csv("../Github/output/03.pull_to_sub_data_382.csv")
sonar_df = pd.read_csv("../Sonar/all_projects_data.csv")
df_extract = []

df_extract = pd.DataFrame({ 'begin_sha': git_df['base.sha'],
                            'end_sha': git_df['merge_commit_sha'],
                            'begin_time': git_df['created_at'],
                            'end_time': git_df['closed_at'],
                            'commits': git_df['commits'],
                            'additions': git_df['additions'],
                            'deletions': git_df['deletions'],
                            'changed_files': git_df['changed_files']
                       })

df_extract['total_time'] = pd.to_datetime(df_extract['end_time']) - pd.to_datetime(df_extract['begin_time'])
time_median = df_extract['total_time'].median()

merged_df_begin_sha = df_extract.merge(sonar_df, left_on='begin_sha', right_on='revision', how='inner')