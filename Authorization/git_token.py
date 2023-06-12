import requests

token = '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'
url = "https://api.github.com/rate_limit"
header = {'Authorization': 'Bearer ' + token}
response = requests.get(url, headers=header)
print(response.json())