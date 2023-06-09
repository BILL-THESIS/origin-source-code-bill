from Authorization import git_token
import requests
import pandas as pd

api_url = 'https://api.github.com/repos/apache/shiro'

data_list = []

response = requests.get(api_url, headers=git_token.header)
response_json = response.json()
response_json_nor = pd.json_normalize(response_json)

df_shiro = pd.DataFrame(response_json_nor)
df_shiro_transpose = df_shiro.transpose()
print(df_shiro_transpose)