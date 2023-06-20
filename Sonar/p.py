import json
from sonarqube import SonarQubeClient
import pandas as pd
import requests


# SonarQubeClien

def search_projects(page_number):
    data_list = []
    # Set up the API request parameters
    params = {
        'ps': 100,  # Number of projects per page
        'p': page_number  # Page number
    }
    headers = {
        'Authorization': 'Bearer sqa_5247f3ea8c7310b315b02b9ac9cba25ef1dac579'  # Replace with your actual API token
    }
    url = 'http://localhost:9000'
    username = "admin"
    password = "admin21"
    sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

    # Send the API request
    response = sonar.projects.search_projects(param=params)

    data_dumps = json.dumps(response)
    data = json.loads(data_dumps)

    # Process the response
    if 'components' in data:
        projects = data['components']
        for project in projects:
            # Process each project as desired
            project_key = project['key']
            project_name = project['name']
            p_qualifier = project['qualifier']
            p_visibility = project['visibility']
            p_lastAnalysisDate = project['lastAnalysisDate']
            p_revision = project['revision']
            # print(f"Project Key: {project_key}, Project Name: {project_name}")

        # Check if there are more pages and recursively call the function for the next page
        if 'paging' in data and 'pageIndex' in data['paging'] and data['paging']['pageIndex'] < data['paging']['total']:
            # print(page_number)
            search_projects(page_number + 1)


# Start the search from the first page
search_projects(1)
df_all = pd.DataFrame()