import requests

token = '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'
url = "https://api.github.com/rate_limit"
header = {'Authorization': 'Bearer ' + token}
response = requests.get(url, headers=header)
print(response.json())