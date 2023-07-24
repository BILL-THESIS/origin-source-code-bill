import pandas as pd
from datetime import timedelta

begin = pd.read_csv('../Sonar/seatunnel_merged_smell_begin.csv')
end = pd.read_csv('../Sonar/seatunnel_merged_smell_end.csv')
git_df = pd.read_csv("../Github/output/02.seatunnel_check_pull.csv")
# =============================================================================
git_ture = git_df.drop(git_df[git_df['merged'] == False].index)
df_extract = pd.DataFrame({
                            'begin_sha': git_ture['base.sha'],
                            'end_sha': git_ture['merge_commit_sha'],
                            'commits': git_ture['commits'],
                            'additions': git_ture['additions'],
                            'deletions': git_ture['deletions'],
                            'changed_files': git_ture['changed_files'],
                            'open_time': git_ture['closed_at'],
                            'closed_time': git_ture['created_at']
                        })

df_extract.set_index(['begin_sha', 'end_sha'])
df_extract['total_time'] = pd.to_datetime(df_extract['open_time']) - pd.to_datetime(df_extract['closed_time'])
# Get the column names as a list
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

df = []
df = pd.DataFrame({
                    'begin_sha': second_m['begin_sha'],
                    'end_sha': second_m['end_sha'],
                    'commits': second_m['commits'],
                    'additions': second_m['additions'],
                    'deletions': second_m['deletions'],
                    'changed_files': second_m['changed_files'],
                    'open_time': second_m['open_time'],
                    'closed_time': second_m['closed_time'],
                    'total_time': second_m['total_time'],
                    'begin_time': second_m['begin_time'],
                    'end_time' : second_m['end_time'],
                    'begin_Dispensables' : second_m['begin_Dispensables'],
                    'begin_Bloaters' : second_m['begin_Change Preventers'],
                    'begin_Change Preventers' : second_m['begin_Change Preventers'],
                    'begin_Couplers' :  second_m['begin_Couplers'],
                    'begin_Object-Orientation Abusers' : second_m['begin_Object-Orientation Abusers'],
                    'end_Dispensables' : second_m['end_Dispensables'],
                    'end_Bloaters' : second_m['end_Change Preventers'],
                    'end_Change Preventers' : second_m['end_Change Preventers'],
                    'end_Couplers' :  second_m['end_Couplers'],
                    'end_Object-Orientation Abusers' : second_m['end_Object-Orientation Abusers'],

})