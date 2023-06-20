import pandas as pd

df = pd.read_csv('../output/03.shiro_label.csv')
df_pull_parents = pd.read_csv('../output/05.shiro.csv')
df_pull_all = pd.read_csv('../output/03.pull_to_sub_data_382.csv')

keyword_base_sha_902 = 'ffde5e8cb99ae70723b40b7e8ea70131773db333'
keyword_base_sha_10 = 'f61c0eb56fd5e09142149755a542e03704c257e8'

filtered_sha_902 = df[df['base.sha'].str.contains(keyword_base_sha_902, case=False)]
filtered_sha_10 = df[df['base.sha'].str.contains(keyword_base_sha_10, case=False)]

merge_draft_false = df_pull_all.loc[df_pull_all['merged'] == False]
merge_draft_true = df_pull_all.loc[df_pull_all['merged'] == True]

merge_sha = merge_draft_true['merge_commit_sha']
base_sha = merge_draft_true['base.sha']

base_sha_drop = base_sha.drop_duplicates()
merge_sha_drop = merge_sha.drop_duplicates()

base_sha_drop.to_csv("shiro_base.txt" , index=False, header=None)
merge_sha_drop.to_csv("merge_base.txt" , index=False, header=None)