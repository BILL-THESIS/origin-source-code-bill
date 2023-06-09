import pandas as pd
import io

repo_data = {
    "repo_1": {"owner": "apache", "name": "shiro"},
    "repo_2": {"owner": "apache", "name": "seatunnel"},
    "repo_3": {"owner": "apache", "name": "jspwiki"},
    "repo_4": {"owner": "apache", "name": "ozone"},
    "repo_5": {"owner": "apache", "name": "cassandra"}
}

df_repo_data = pd.DataFrame(repo_data)

def repo(data):
    for repo_key, repo_value in repo_data.items():
        # list_obj = []
        repo_owner = repo_value["owner"]
        repo_name = repo_value["name"]
        api_pulls = "https://api.github.com/repos/" + repo_owner + "/" + repo_name + "/pulls?state=closed"
        print(api_pulls)


api = repo(repo_data)
print(api)


obj = []
df = pd.DataFrame(obj)

df['repo_name'] = [
    'https://api.github.com/repos/apache/shiro/pulls?state=closed',
    'https://api.github.com/repos/apache/seatunnel/pulls?state=closed',
    'https://api.github.com/repos/apache/jspwiki/pulls?state=closed',
    'https://api.github.com/repos/apache/ozone/pulls?state=closed',
    'https://api.github.com/repos/apache/cassandra/pulls?state=closed'
]

api_pull_shiro = df['repo_name'][0]
api_pull_seatunnel = df['repo_name'][1]
api_pull_jspwiki = df['repo_name'][2]
api_pull_ozone = df['repo_name'][3]
api_pull_cassandra = df['repo_name'][4]
