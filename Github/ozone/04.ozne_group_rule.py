import pandas as pd

df =pd.read_csv("../../Sonar/output/all_smells_test_2.csv")
df2 = pd.read_csv("../../Sonar/output/all_smells_test_3.csv")
df3 = pd.read_csv("../../Sonar/output/all_smells_1834.csv")
df4 = pd.read_csv("../../Sonar/output/all_smells.csv")

df_concat =pd.concat([df,df2,df3,df4])
grouped_data_dropna = df_concat.dropna(axis='columns')
set_index_df = grouped_data_dropna.set_index(['project', 'rule'])
set_index_df.drop_duplicates()
rule_counts = set_index_df.groupby(['project', 'rule']).size()
rule_df = pd.DataFrame(rule_counts)
pivot_df = rule_df.pivot_table(index='project', columns='rule', fill_value=0)
pivot_df.to_csv("smells_all.csv", index='project')