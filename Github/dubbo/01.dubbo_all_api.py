import requests
from Authorization import git_token
from Repository_git import repo
import pandas as pd

url = 'https://api.github.com/repos/apache/dubbo/pulls?state=closed&per_page=300&page='

def cell_api(api, fromPage=1, toPage=5, **page):
    data = page.get('page', [])
    for i in range(fromPage, toPage + 1, 1):
        cell = requests.get(api + str(i), headers=git_token.header)
        data += cell.json()
        print('loading page' + str(i))
    if len(data) > 10:
        return (data)


data_list_api = cell_api(url, fromPage=1, toPage=200)
df = pd.json_normalize(data_list_api)
df.to_csv("D:\origin-source-code-bill\Github\output\dubbo_all_api.csv")