import pandas as pd
import json
from sonarqube import SonarQubeClient

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
    response = sonar.projects.search_projects(**params)

    data = response['components']

    # Process the response
    if data:
        df_list = []
        for project in data:
            df_json = json.dumps(project)
            obj = json.loads(df_json)
            obj_json_nor = pd.json_normalize(obj)
            df = pd.DataFrame(obj_json_nor)
            df_list.append(df)

        df_result = pd.concat(df_list)
        print(df_result)
        df_result.to_csv("all.csv", index=False)

        if 'paging' in response:
            total_pages = response['paging']['total']
            current_page = response['paging']['pageIndex']
            if current_page < total_pages:
                search_projects(page_number + 1)
                print("Loading page", page_number)

# Start the search from the first page
search_projects(1)
