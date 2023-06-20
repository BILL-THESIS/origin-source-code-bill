import json
from sonarqube import SonarQubeClient
import pandas as pd
import requests
# SonarQubeClien

def search_projects(page_number):
    # Set up the API request parameters
    params = {
        'ps': 500,  # Number of projects per page
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
            df_json = json.dumps(project)
            obj = json.loads(df_json)
            obj_json_nor = pd.json_normalize(obj)
            print(obj_json_nor)
            obj_json_nor.to_csv("Sonar_all_projects.csv")

        # Check if there are more pages and recursively call the function for the next page
        if 'paging' in data and 'pageIndex' in data['paging'] and data['paging']['pageIndex'] < data['paging']['total']:
            # print(page_number)
            search_projects(page_number + 1)

# Start the search from the first page
search_projects(1)
