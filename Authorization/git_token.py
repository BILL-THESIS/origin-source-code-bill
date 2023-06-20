import requests

token = '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'
url = "https://api.github.com/rate_limit"
header = {'Authorization': 'Bearer ' + token}
response = requests.get(url, headers=header)
print(response.json())