import pandas as pd
import sonarqube
from sonarqube import SonarQubeClient
import json

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

params = {
    # 'componentKeys': key,
    # 'rules': rule,
    'scopes' : 'MAIN',
    'types' : 'CODE_SMELL',
    'ps' : 500
    # "pageSize" : 500
    # 'paging.paegSize': 500,  # Number of projects per page
    # 'paging.paeIndex': page_number  # Page number
}

issues = sonar.issues.search_issues(param=params)
data_dumps = json.dumps(issues)
data = json.loads(data_dumps)
df_is= pd.json_normalize(data)

data_list = []
for i in data:
    issues_1  = data['components']
    issues_2 = data['issues']
    df = pd.DataFrame(issues_1)
    df2 = pd.DataFrame(issues_2)
    df3 = data_list.append(df2)
    print(data)