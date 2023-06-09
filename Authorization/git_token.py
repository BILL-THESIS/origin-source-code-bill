import requests

token = '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'
url = "https://api.github.com/rate_limit"
header = {'Authorization': 'Bearer ' + token}
response = requests.get(url, headers=header)
print(response.json())