import pandas as pd

data = [
    {'label': 'apache:1.11.x', 'ref': '1.11.x', 'sha': 'ffde5e8cb99ae70723b40b7e8ea70131773db333', 'user': {'login': 'apache', 'id': 47359, 'node_id': 'MDEyOk9yZ2FuaXphdGlvbjQ3MzU5', 'avatar_url': 'https://avatars.githubusercontent.com/u/47359?v=4', 'gravatar_id': '', 'url': 'https://api.github.com/users/apache'}},
    {'label': 'apache:1.11.x', 'ref': '1.11.x', 'sha': 'b7e8ea70131773', 'user': {'login': 'apache', 'id': 47358, 'node_id': 'MDEQ3MzU5', 'avatar_url': 'https://avatars.githubusercontent.com/u/47358?v=4', 'gravatar_id': '', 'url': 'https://api.github.com/users/apache'}}
]

df = pd.DataFrame(data)
print(df)

login_values = df['user'].apply(lambda x: x['login'])
print(login_values)

