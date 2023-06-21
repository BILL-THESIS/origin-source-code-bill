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
        df_list = []
        for project in projects:

            df_json = json.dumps(project)
            obj = json.loads(df_json)
            obj_json_nor = pd.json_normalize(obj)
            df = pd.DataFrame(obj_json_nor)
            df_list.append(df)
            print(df_list)

            df_result = pd.concat(df_list)
            print(df_result)


            if 'paging' in data and 'pageIndex' in data['paging'] and data['paging']['pageIndex'] < data['paging']['total']:
                search_projects(page_number + 1)
                print("loading", page_number)

            output_file = f"page_{page_number}.csv"
            df_result.to_csv(output_file, index=False)

            # if 'paging' in response:
            #     total_pages = response['paging']['total']
            #     current_page = response['paging']['pageIndex']
            #     if current_page < total_pages:
            #         search_projects(page_number + 1)
            #         print("Loading page", page_number)

search_projects(1)
