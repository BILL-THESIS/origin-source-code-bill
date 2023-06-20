import requests
import pandas as pd
from Authorization import git_token


api_get = 'https://api.github.com/repos/apache/seatunnel/pulls?state=closed&per_page=300&page='


def cell_api(api, fromPage=1, toPage=5, **page):
    data = page.get('page', [])
    for i in range(fromPage, toPage + 1, 1):
        cell = requests.get(api + str(i), headers=git_token.header)
        data += cell.json()
        print('loading page' + str(i))
    if len(data) > 10:
        return (data)


data_list_api = cell_api(api_get, fromPage=1, toPage=50)
df = pd.json_normalize(data_list_api)
df.to_csv("D:\origin-source-code-bill\Github\output\01.csv")
