import pandas as pd

g = pd.read_csv('../models/group_smells.csv')
a = pd.read_csv('seatunnel_smells_all.csv')
rule_key_smell = a.sum()
sum = pd.read_csv('../Sonar/rule_key_smell.csv')

df =[]


merged_df_Bloaters = pd.merge(g, sum,  left_on='Bloaters', right_on='key', how='inner')
merged_df_Bloaters.drop(columns=['Change Preventers', 'Couplers', 'Dispensables', 'Object-Orientation Abusers'], axis=1)

merged_df_Change_Preventers = pd.merge(g, sum,  left_on='Change Preventers', right_on='key', how='inner')
merged_df_Change_Preventers.dropna()

merged_df_Couplers = pd.merge(g, sum,  left_on='Couplers', right_on='key', how='inner')
merged_df_Couplers.dropna()

merged_df_Dispensables = pd.merge(g, sum,  left_on='Dispensables', right_on='key', how='inner')
merged_df_Dispensables.dropna()

merged_df_Object_Orientation_Abusers = pd.merge(g, sum,  left_on='Object-Orientation Abusers', right_on='key', how='inner')
merged_df_Object_Orientation_Abusers.dropna()

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

# pivot_df with sonar_smells
df_transposed_begin = begin.transpose()
begin_drop = begin.drop(columns=['sha','key','time', 'revision'])
# df_transposed_begin.columns = df_transposed_begin.iloc[0]
# df_transposed_begin = df_transposed_begin[1:]

df_transposed_end = end.transpose()
# df_transposed_end.columns = df_transposed_end.iloc[0]
# df_transposed_end = df_transposed_end[1:]

# check merge df 1 / df 2 left and right
join_smells_d_begin = pivot_df_d.merge(df_transposed_begin, left_index=True, right_index=True)
join_smells_d_drop_begin = join_smells_d_begin.drop(columns='Dispensables')
d_begin = join_smells_d_drop_begin.T
d_begin['Dispensables'] = d_begin.sum(axis=1)

join_smells_b_begin = pivot_df_b.merge(df_transposed_begin , left_index=True, right_index=True)
join_smells_b_drop_begin = join_smells_b_begin.drop(columns='Bloaters')
b_begin = join_smells_b_drop_begin.T
b_begin['Bloaters'] = b_begin.sum(axis=1)

join_smells_cp_begin = pivot_df_cp.merge(df_transposed_begin , left_index=True, right_index=True)
join_smells_cp_drop = join_smells_cp_begin.drop(columns='Change Preventers')
cp_begin = join_smells_cp_drop.T
cp_begin['Change Preventers'] = cp_begin.sum(axis=1)

join_smells_c_begin = pivot_df_c.merge(df_transposed_begin , left_index=True, right_index=True)
join_smells_c_drop = join_smells_c_begin.drop(columns='Couplers')
c_begin = join_smells_c_drop.T
c_begin['Couplers'] = c_begin.sum(axis=1)

join_smells_oop_begin = pivot_df_oop.merge(df_transposed_begin , left_index=True, right_index=True)
join_smells_oop_drop = join_smells_oop_begin.drop(columns='Object-Orientation Abusers')
oop_begin = join_smells_c_drop.T
oop_begin['Object-Orientation Abusers'] = oop_begin.sum(axis=1)

merged_smell_begin = pd.concat([begin, d_begin['Dispensables'], b_begin['Bloaters'], cp_begin['Change Preventers'], c_begin['Couplers'], oop_begin['Object-Orientation Abusers']], axis=1).fillna(0)

# check merge df 1 / df 2 left and right
join_smells_d_end = pivot_df_d.merge(df_transposed_end, left_index=True, right_index=True)
join_smells_d_drop_end = join_smells_d_end.drop(columns='Dispensables')
d_end = join_smells_d_drop_end.T
d_end['Dispensables'] = d_end.sum(axis=1)

join_smells_b_end = pivot_df_b.merge(df_transposed_end , left_index=True, right_index=True)
join_smells_b_drop_end = join_smells_b_end.drop(columns='Bloaters')
b_end = join_smells_b_drop_end.T
b_end['Bloaters'] = b_end.sum(axis=1)

join_smells_cp_end = pivot_df_cp.merge(df_transposed_end , left_index=True, right_index=True)
join_smells_cp_drop_end = join_smells_cp_end.drop(columns='Change Preventers')
cp_end = join_smells_cp_drop_end.T
cp_end['Change Preventers'] = cp_end.sum(axis=1)

join_smells_c_end = pivot_df_c.merge(df_transposed_end , left_index=True, right_index=True)
join_smells_c_drop_end = join_smells_c_end.drop(columns='Couplers')
c_end = join_smells_c_drop_end.T
c_end['Couplers'] = c_end.sum(axis=1)

join_smells_oop_end = pivot_df_oop.merge(df_transposed_end , left_index=True, right_index=True)
join_smells_oop_drop_end = join_smells_oop_end.drop(columns='Object-Orientation Abusers')
oop_end = join_smells_oop_drop_end.T
oop_end['Object-Orientation Abusers'] = oop_end.sum(axis=1)

merged_smell_end = pd.concat([begin, d_end['Dispensables'], b_end['Bloaters'], cp_end['Change Preventers'], c_end['Couplers'], oop_end['Object-Orientation Abusers']], axis=1).fillna(0)