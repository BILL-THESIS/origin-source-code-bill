import requests

token = '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'
url = "https://api.github.com/rate_limit"
header = {'Authorization': 'Bearer ' + token}
response = requests.get(url, headers=header)
print(response.json())