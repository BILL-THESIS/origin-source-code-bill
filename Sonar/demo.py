import pandas as pd

# Sample df_rules DataFrame
df_rules = pd.DataFrame({
    'project': ['P1', 'P1', 'P2', 'P2', 'P3'],
    'A': [10, 15, 5, 20, 0],
    'B': [5, 10, 15, 0, 10],
    'C': [20, 25, 10, 5, 30]
})

# Sample melt_df DataFrame
melt_df = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'category': ['Cat1', 'Cat2', 'Cat3']
})

# Melting the df_rules DataFrame to long format
df_rules_melted = df_rules.melt(id_vars=['project'], var_name='rule', value_name='count')

# Merging the melted df_rules DataFrame with melt_df
merged_df = df_rules_melted.merge(melt_df, how='left', left_on='rule', right_on='key')

# Pivoting the merged DataFrame to get the desired format
result_df = merged_df.pivot_table(index='project', columns='category', values='count', aggfunc='sum').reset_index()

# Display the result DataFrame
print(result_df)
