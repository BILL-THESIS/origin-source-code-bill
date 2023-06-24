import requests

token = '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'
url = "https://api.github.com/rate_limit"
header = {'Authorization': 'Bearer ' + token}
response = requests.get(url, headers=header)
print(response.json())