import pandas as pd
from datetime import timedelta

git_df = pd.read_csv("../Github/output/02.seatunnel_check_pull.csv")
sonar_project_key = pd.read_csv("../Sonar/output/all_projects_data.csv")
sonar_smells = pd.read_csv("../Sonar/output/smells_all_seatunnel.csv")

git_ture = git_df.drop(git_df[git_df['merged'] == False].index)

df_extract = []
sonar_key_set = []

sonar_key_set = pd.DataFrame({
                                'key': sonar_project_key['key'],
                                'revision': sonar_project_key['revision']
                            })

df_extract_begin = pd.DataFrame({
                                    'begin_sha': git_ture['base.sha'],
                                    'begin_time': git_ture['created_at']
                                })

df_extract_end = pd.DataFrame({
                                'end_sha': git_ture['merge_commit_sha'],
                                'end_time': git_ture['closed_at']
                            })

df_extract = pd.DataFrame({
                            'begin_sha': git_ture['base.sha'],
                            'end_sha': git_ture['merge_commit_sha'],
                            'commits': git_ture['commits'],
                            'additions': git_ture['additions'],
                            'deletions': git_ture['deletions'],
                            'changed_files': git_ture['changed_files']
                        })

df_extract['total_time'] = pd.to_datetime(df_extract_end['end_time']) - pd.to_datetime(df_extract_begin['begin_time'])
time_median = df_extract['total_time'].median()

merged_df_begin = df_extract.merge(sonar_key_set, left_on='begin_sha', right_on='revision')
merged_df_end = df_extract.merge(sonar_key_set, left_on='end_sha', right_on='revision')

