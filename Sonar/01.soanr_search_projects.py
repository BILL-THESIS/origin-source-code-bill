import json
import pandas as pd
from sonarqube import SonarQubeClient

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

params = {
    'ps': 500,
    'p': 1
}

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

data_list = []

# Step 1: Loop through the API pages for Sonar project keys
while True:
    projects = sonar.projects.search_projects(**params)
    data = projects['components']

    # Break the loop if no more projects to retrieve
    if not data:
        break

    # Step 2: Save data from API projects key to data list
    for project in data:
        # data_list.append(project['key'])
        data_list.append(project)
        print(data_list)

    if 'paging' in projects and projects['paging']['pageIndex'] < projects['paging']['total']:
        params['p'] += 1  # Increment the page index
        print(projects['paging']['pageIndex'])
    else:

        break

# Step 3: Concatenate all data lists from different pages
df_result = pd.concat([pd.DataFrame(data_list, columns=['project_key'])])
print(df_result)

# Step 4: Export data list to a CSV file
df_result.to_csv("Sonar_api.csv", index=False)