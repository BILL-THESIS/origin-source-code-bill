import requests
from Authorization import git_token

import pandas as pd

url = 'https://api.github.com/repos/apache/shiro/pulls?state=closed&per_page=300&page='
data = ('page', [])
page = 1

while url():
    response = requests.get(url, headers= git_token.header)
    if response.status_code == 200:
        data += response.json()
        # page_data = response.json()

        # Check if there is a next page
        link_header = response.headers.get('Link', '')
        if 'rel="next"' in link_header:
            page += 1
            print("::::::::::::::::::", page)
            url = f'https://api.github.com/repos/apache/shiro/pulls?state=closed&per_page=300&page={page}'
        else:
            url = None
    else:
        print('Error:', response.status_code)
        break

print(data)


df = pd.DataFrame(data)
