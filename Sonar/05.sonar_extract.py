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
first_m = df_extract.merge(begin, left_on=['begin_sha', 'end_sha'], right_on=['begin_begin_sha', 'begin_end_sha'])
first_m.drop_duplicates()
second_m = df_extract.merge(end , left_on=['begin_sha', 'end_sha'], right_on=['end_begin_sha', 'end_end_sha'])
second_m.drop_duplicates()

all_df = first_m.merge(second_m, on=['begin_sha', 'end_sha'])

df = []
df = pd.DataFrame({
                    'begin_sha': all_df['begin_sha'],
                    'end_sha': all_df['end_sha'],
                    'commits': all_df['commits_x'],
                    'additions': all_df['additions_x'],
                    'deletions': all_df['deletions_x'],
                    'changed_files': all_df['changed_files_x'],
                    # 'open_time': all_df['open_time_x'],
                    # 'closed_time': all_df['closed_time_x'],
                    'total_time': all_df['total_time_x'],
                    'begin_time': all_df['begin_begin_time'],
                    'end_time' : all_df['end_end_time'],
                    'begin_Dispensables' : all_df['begin_Dispensables'],
                    'begin_Bloaters' : all_df['begin_Change Preventers'],
                    'begin_Change Preventers' : all_df['begin_Change Preventers'],
                    'begin_Couplers' :  all_df['begin_Couplers'],
                    'begin_Object-Orientation Abusers' : all_df['begin_Object-Orientation Abusers'],
                    'end_Dispensables' : all_df['end_Dispensables'],
                    'end_Bloaters' : all_df['end_Change Preventers'],
                    'end_Change Preventers' : all_df['end_Change Preventers'],
                    'end_Couplers' :  all_df['end_Couplers'],
                    'end_Object-Orientation Abusers' : all_df['end_Object-Orientation Abusers'],

})
