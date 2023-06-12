import pandas as pd
import requests
from Authorization import git_token

df = pd.read_csv("01.shiro_all_pll.csv")
df2 = pd.read_csv("02.shiro_api_pulls.csv")


data_list = []

for i in df['url']:
    data = requests.get(i, headers=git_token.header)
    data_list.append(data.json())
    print("========= :" , data_list)
    obj = pd.json_normalize(data_list)
    print(obj)

df3 = pd.DataFrame(obj)
print(df)


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