import pandas as pd

df =pd.read_csv("smells_all.csv")
grouped_data_dropna = df.dropna(axis='columns')
# set_index_df = grouped_data_dropna.set_index(['project', 'rule'])
# rule_counts = set_index_df.groupby(['project', 'rule']).size()
# rule_df = pd.DataFrame(rule_counts)
# pivot_df = rule_df.pivot_table(index='project', columns='rule', fill_value=0)
# pivot_df.to_csv("smells_all_seatunnel.csv", index='project')