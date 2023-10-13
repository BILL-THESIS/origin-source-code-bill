import pandas as pd

sonar_group_smells = pd.read_csv('../../Sonar/output/group_smells.csv')
x= pd.read_csv('../../Sonar/output/group_smells.csv')
# sonar_project_key = pd.read_csv("../Sonar/output/all_projects_data.csv")
sonar_smells = pd.read_csv("../../Sonar/output/smell_test_data.csv")

df_extract_begin = pd.read_csv('ozone_smells_begin.csv')
df_extract_end = pd.read_csv('ozone_smells_end.csv')

begin = df_extract_begin
end = df_extract_end

# pivot_df with sonar_smells
df_transposed_begin = begin.transpose()
# begin_drop = begin.drop(columns=['sha','key','time', 'revision'])
# df_transposed_begin.columns = df_transposed_begin.iloc[0]
# df_transposed_begin = df_transposed_begin[1:]

df_transposed_end = end.transpose()
# df_transposed_end.columns = df_transposed_end.iloc[0]
# df_transposed_end = df_transposed_end[1:]

pivot_df_d_1 = pd.merge(sonar_smells,sonar_group_smells ,left_on='key', right_on='Dispensables').set_index('key')
pivot_df_d = pivot_df_d_1.drop(columns=['name', 'Smells','Bloaters','Object-Orientation Abusers','Couplers','Change Preventers'])

pivot_df_b_1 = pd.merge(sonar_smells,sonar_group_smells ,left_on='key', right_on='Bloaters').set_index('key')
pivot_df_b = pivot_df_b_1.drop(columns=['name', 'Smells','Dispensables','Object-Orientation Abusers','Couplers','Change Preventers'])

pivot_df_cp_1 = pd.merge(sonar_smells,sonar_group_smells ,left_on='key', right_on='Change Preventers').set_index('key')
pivot_df_cp = pivot_df_cp_1.drop(columns=['name', 'Smells','Dispensables','Object-Orientation Abusers','Couplers','Bloaters'])

pivot_df_c_1 = pd.merge(sonar_smells,sonar_group_smells ,left_on='key', right_on='Couplers').set_index('key')
pivot_df_c = pivot_df_c_1.drop(columns=['name', 'Smells','Dispensables','Object-Orientation Abusers','Change Preventers','Bloaters'])

pivot_df_oop_1 =  pd.merge(sonar_smells,sonar_group_smells ,left_on='key', right_on='Object-Orientation Abusers').set_index('key')
pivot_df_oop = pivot_df_oop_1.drop(columns=['name', 'Smells','Dispensables','Couplers','Change Preventers','Bloaters'])

# check merge df 1 / df 2 left and right
join_smells_d_begin = pivot_df_d.merge(df_transposed_begin, left_index=True, right_index=True)
join_smells_d_drop_begin = join_smells_d_begin.drop(columns='Dispensables')
d_begin = join_smells_d_begin.T
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

# merged_smell_begin.to_parquet('../output/seatunnel_merged_smell_begin.parquet', index=False)
# merged_smell_end.to_parquet('../output/seatunnel_merged_smell_end.parquet', index=False)