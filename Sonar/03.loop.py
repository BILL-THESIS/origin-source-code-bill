import json
from sonarqube import SonarQubeClient
import pandas as pd

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

params = {
    'ps': 500,
    'p': 1
}

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

df_list = []

while True:
    projects = sonar.projects.search_projects(**params)
    data = projects['components']

    for project in data:
        df_json = json.dumps(project)
        obj = json.loads(df_json)
        obj_json_nor = pd.json_normalize(obj)
        df = pd.DataFrame(obj_json_nor)
        df_list.append(df)

    # if 'paging' in data and data['paging']['pageIndex'] < data['paging']['totalPages']:
    #     params['p'] += 1  # Increment the page index
    if 'paging' in projects:
        total_pages = projects['paging']['total']
        current_page = projects['paging']['pageIndex']
        if current_page < total_pages:
            params['p'] += 1
            print("Loading page",params['p'] )
    else:
        break

df_result = pd.concat(df_list)
print(df_result)
df_result.to_csv("all_2.csv", index=False)
