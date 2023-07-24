import pandas as pd
from datetime import timedelta

git_df = pd.read_csv("../../Github/output/02.seatunnel_check_pull.csv")
sonar_project_key = pd.read_csv("../output/all_projects_data.csv")
sonar_smells = pd.read_csv("../Sonar/output/smells_all_seatunnel.csv")

sonar_smells.set_index(['key'])

git_ture = git_df.drop(git_df[git_df['merged'] == False].index)

df_extract = []
sonar_key_set = []

sonar_key_set = pd.DataFrame({
                                'key': sonar_project_key['key'],
                                'revision': sonar_project_key['revision']
                            })
sonar_project_key.set_index(['revision'])

df_extract_begin = pd.DataFrame({
                                    'sha': git_ture['base.sha'],
                                    'time': git_ture['created_at']
                                })
df_extract_begin.set_index(['sha'])

df_extract_end = pd.DataFrame({
                                'sha': git_ture['merge_commit_sha'],
                                'time': git_ture['closed_at']
                            })
df_extract_end.set_index(['sha'])

df_extract = pd.DataFrame({
                            'begin_sha': git_ture['base.sha'],
                            'end_sha': git_ture['merge_commit_sha'],
                            'commits': git_ture['commits'],
                            'additions': git_ture['additions'],
                            'deletions': git_ture['deletions'],
                            'changed_files': git_ture['changed_files']
                        })

df_extract.set_index(['begin_sha', 'end_sha'])
df_extract['total_time'] = pd.to_datetime(df_extract_end['time']) - pd.to_datetime(df_extract_begin['time'])
time_median = df_extract['total_time'].median()

merged_df_begin = df_extract_begin.merge(sonar_key_set, left_on='sha', right_on='revision')
merged_df_end = df_extract_end.merge(sonar_key_set, left_on='sha', right_on='revision')

begin = merged_df_begin.merge(sonar_smells, on='key')
end = merged_df_end.merge(sonar_smells , on='key')

# # Get the column names as a list
columns_begin = begin.columns.tolist()
columns_end = end.columns.tolist()

# Add the prefix to each column name using a list comprehension
columns_with_prefix_begin = ['begin_' + col for col in columns_begin]
columns_with_prefix_end = ['end_' + col for col in columns_end]

# Assign the modified column names back to the DataFrame
begin.columns = columns_with_prefix_begin
end.columns = columns_with_prefix_end

# data Frames Extract GitHub with Begin
# merged_df = df_extract.merge(begin, on='begin_sha')

first_m = df_extract.merge(begin, on='begin_sha')
first_m.drop_duplicates()
second_m = first_m.merge(end , on='end_sha')
second_m.drop_duplicates()