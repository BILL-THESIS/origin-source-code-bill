import pandas as pd

df_smells = pd.read_csv('../../Sonar/output/smells_all.csv')
df_projects_sonar = pd.read_csv('../../Sonar/output/all_projects_data.csv')
df_git = pd.read_csv('ozone_check_pull_1.csv')

git_ture = df_git.drop(df_git[df_git['merged'] == False].index)

df_extract = pd.DataFrame({
                            'url' : git_ture['url'],
                            'begin_sha': git_ture['base.sha'],
                            'end_sha': git_ture['merge_commit_sha'],
                            'commits': git_ture['commits'],
                            'additions': git_ture['additions'],
                            'deletions': git_ture['deletions'],
                            'changed_files': git_ture['changed_files']
                        })

df = pd.merge(df_smells , df_projects_sonar , left_on='project', right_on='key')

merged_df_begin = df_extract.merge(df, left_on='begin_sha', right_on='revision')
merged_df_end = df_extract.merge(df, left_on='end_sha', right_on='revision')
