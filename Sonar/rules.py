import sonarqube
import json
import pandas as pd
from sonarqube import SonarQubeClient

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

params = {
    'languages': 'java',
    'types': 'CODE_SMELL',
    'ps': 500,
    'p': 1
    # 'tags': 'correctness'
}

rules = sonar.rules.search_rules(**params)
data_dumps = json.dumps(rules)
data = json.loads(data_dumps)

if 'rules' in data:
    rules_java = data['rules']
    df_json = json.dumps(rules_java)
    obj = json.loads(df_json)
    obj_json_nor = pd.json_normalize(obj)

if 'paging' in data and 'pageIndex' in data['paging'] and data['paging']['pageIndex'] < data['paging']['total']:
    params['p'] += 1
    print("loading" ,params['p'])

