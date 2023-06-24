import pandas as pd

df = pd.read_csv("ozone_check_pull_1.csv")


label = df[['url' , 'id', 'node_id', 'number',
            'state', 'created_at', 'updated_at',
            'closed_at', 'merged_at', 'merge_commit_sha',
            'commits', 'additions', 'deletions',
            'changed_files',
            'user.login', 'user.id',
            'user.type', 'head.sha', 'head.user.login',
            'head.user.id', 'head.user.type', 'head.repo.id',
            'base.label', 'base.sha', 'base.repo.created_at',
            'base.repo.updated_at', 'base.repo.pushed_at']]

merge_draft_false = df.loc[df['merged'] == False]
merge_draft_true = df.loc[df['merged'] == True]

merge_sha = merge_draft_true['merge_commit_sha']
base_sha = merge_draft_true['base.sha']

base_sha_drop = base_sha.drop_duplicates()
merge_sha_drop = merge_sha.drop_duplicates()

base_sha_drop.to_csv("ozone_base.txt" , index=False, header=None)
merge_sha_drop.to_csv("ozone_merge.txt" , index=False, header=None)