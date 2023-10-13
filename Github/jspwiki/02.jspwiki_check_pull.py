import pandas as pd
import requests
from Authorization import git_token

df = pd.read_csv("../output/jspwiki.csv")
data_list = []

df3 = pd.read_csv('jspwiki_check_pull.csv')

label = df3[['url' , 'id', 'node_id', 'number',
            'state', 'created_at', 'updated_at',
            'closed_at', 'merged_at', 'merge_commit_sha',
            'commits', 'additions', 'deletions',
            'changed_files',
            'user.login', 'user.id',
            'user.type', 'head.sha', 'head.user.login',
            'head.user.id', 'head.user.type', 'head.repo.id',
            'base.label', 'base.sha', 'base.repo.created_at',
            'base.repo.updated_at', 'base.repo.pushed_at']]

df3.to_csv("seatunnel_check_pull.csv" , index=False)

merge_draft_false = df3.loc[df3['merged'] == False]
merge_draft_true = df3.loc[df3['merged'] == True]

merge_sha = merge_draft_true['merge_commit_sha']
base_sha = merge_draft_true['base.sha']

base_sha_drop = base_sha.drop_duplicates()
merge_sha_drop = merge_sha.drop_duplicates()

base_sha_drop.to_csv("_base.txt" , index=False, header=None)
merge_sha_drop.to_csv("merge_base.txt" , index=False, header=None)