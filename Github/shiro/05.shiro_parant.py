import pandas as pd
import requests
from Authorization import git_token
import requests_cache

df_bash_sha =pd.read_csv('../output/shiro_bash_sha.csv')
df = pd.read_csv('../output/03.pull_to_sub_data_382.csv')


data_odj = []
list_all_pd = []
for i in df['commits_url']:
    print(i)
    response_url = requests.get(i, headers=git_token.header)
    # print('New response cached')
    data_odj += response_url.json()
    # print("OBJ:", data_odj)
    data_loads_json = pd.json_normalize(data_odj)
    # print("data_loads_json:", data_loads_json)

    data_loads_json['index'] = i
    data_loads_json.set_index(['index', 'sha', 'node_id', 'commit.tree.sha'])
    append_obj = list_all_pd.append(data_loads_json)
    all_pd = pd.concat(list_all_pd)
    drop_all_pd = all_pd.drop_duplicates(subset=['sha', 'node_id'])
    set_pull = drop_all_pd.set_index(['index', 'sha'])
    # set_pull.to_csv('get_pull_commits_all.csv')

    if i == df['commits_url'][862]:
        break


data_parant_pall = pd.DataFrame(set_pull)
print(data_parant_pall)
data_parant_pall.to_csv('05.shiro.csv')
