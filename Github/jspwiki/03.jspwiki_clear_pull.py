import pandas as pd

df = pd.read_csv('jspwiki_check_pull.csv')

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


git_ture = df.drop(df[df['merged'] == False].index)