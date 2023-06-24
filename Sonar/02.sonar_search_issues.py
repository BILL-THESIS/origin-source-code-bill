import pandas as pd
import requests
from sonarqube import SonarQubeClient
import json

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

project_keys = pd.read_csv("all_projects.csv")

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

all_issues = []

for project_key in project_keys['project_key']:
    params = {
        'componentKeys': project_key,
        'scopes': 'MAIN',
        'types': 'CODE_SMELL',
        'ps': 500,
        'p': 1
    }

    print(project_key)
    while True:
        response = sonar.issues.search_issues(**params)
        issues = response['issues']

        all_issues.extend(issues)
        # print(all_issues)

        if len(issues) < params['ps']:
            break

        params['p'] += 1

        if not issues:  # Check if the data page is empty
            break

        data_list = []

        for issue in all_issues:
            # Extract the required data from the issue object
            data = {
                'key': issue['key'],
                'type': issue['type'],
                # Add more fields as needed
            }

            data_list.append(data)
            # print(data_list)

        # Concatenate the data_page with the previous data
        data_list.extend(issues)

    grouped_data = pd.DataFrame(data_list).groupby('key')