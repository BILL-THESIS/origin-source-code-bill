import pandas as pd
import requests
from sonarqube import SonarQubeClient
import json

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

# project_keys = pd.read_csv("all_projects.csv")
project_keys = pd.read_csv("Sonar_api.csv")

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

all_issues = []

for project_key in project_keys['key']:
    params = {
        'componentKeys': project_key,
        'scopes': 'MAIN',
        'languages': 'java' ,
        'types': 'CODE_SMELL',
        'ps': 500,
        'p': 1
    }

    print(project_key)

    while True:
        response = sonar.issues.search_issues(**params)
        issues = response['issues']

        all_issues.extend(issues)

        if len(issues) < params['ps']:
            break

        params['p'] += 1

        if not issues:  # Check if the data page is empty
            break

        all_issues_df = pd.DataFrame(all_issues)
        print(all_issues_df)

        data_list = []

        for issue in all_issues_df:
            # Extract the required data from the issue object
            data = {
                'key': issue['key'],
                'type': issue['type'],
                'rule': issue['rule'],
                'project': issue['project'],
                'effort': issue['effort'],
                'debt': issue['debt']
            }

            data_list_append = data_list.append(data)
            # print(data_list_append)

        # Concatenate the data_page with the previous data
        data_list_extend = data_list.extend(issues)

    grouped_data = pd.DataFrame(data_list_append)
    grouped_data_dropna = grouped_data.dropna(axis='columns')
    set_index_df = grouped_data_dropna.set_index(['project', 'rule'])
    rule_counts = set_index_df.groupby(['project', 'rule']).size()
    rule_df = pd.DataFrame(rule_counts)
    pivot_df = rule_df.pivot_table(index='project', columns='rule', fill_value=0)
    pivot_df.to_csv("smells_all_24112023.csv", index='project')
