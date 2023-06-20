import pandas as pd

df_pull = pd.read_csv('../output/03.pull_to_sub_data_382.csv')
df_pull_parents = pd.read_csv('../output/05.shiro.csv')

# Find keyword in the 'name' column
keyword_base_sha = 'f61c0eb56fd5e09142149755a542e03704c257e8'
keyword_parents = 'cbdf317007f991f2d3f8d95b8cbaad9ba92c418a'

filtered_sha = df_pull[df_pull['base.sha'].str.contains(keyword_base_sha, case=False)]
filtered_parents = df_pull_parents[df_pull_parents['index'].str.contains(keyword_parents, case=False)]

label = df_pull_parents[['sha', "parents"]]

structured_array = label.to_records(index=True)

# df = pd.DataFrame()
# x = label.values.tolist()
# for sublist in structured_array:
#     for dictionary in sublist:
#         flattened_data = pd.json_normalize(dictionary,'parents',['sha'])
#         df = df.append(flattened_data, ignore_index=True)
# print(df)
