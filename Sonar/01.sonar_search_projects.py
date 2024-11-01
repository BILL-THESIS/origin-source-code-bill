import json
import pandas as pd
from sonarqube import SonarQubeClient
from Authorization import sonar_token


# Constants
PAGE_SIZE = 500
PKL_FILE_NAME = "output/01.sonar_search_pull.pkl"


def get_sonar_client(url, username, password):
    """Initialize and return a SonarQubeClient."""
    return SonarQubeClient(sonarqube_url=url, username=username, password=password)


def fetch_all_projects(sonar_client, page_size):
    """Fetch all projects from SonarQube API."""
    params = {
        'ps': page_size,
        'p': 1
    }
    projects_list = []

    try:
        while True:
            response = sonar_client.projects.search_projects(**params)
            projects = response.get('components', [])

            if not projects:
                break

            projects_list.extend(projects)

            if 'paging' in response and response['paging']['pageIndex'] < response['paging']['total']:
                params['p'] += 1
            else:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    return projects_list


def projects_to_dataframe(projects_list):
    """Convert list of project dictionaries to a DataFrame."""
    return pd.DataFrame(projects_list)


def save_to_csv(df, file_name):
    """Save DataFrame to a PKL file."""
    df.to_pickle(file_name)
    print(f"Data successfully exported to {file_name}")


def main():
    sonar_client = get_sonar_client(sonar_token.URL, sonar_token.USERNAME, sonar_token.PASSWORD)
    projects_list = fetch_all_projects(sonar_client, PAGE_SIZE)
    df_projects = projects_to_dataframe(projects_list)
    save_to_csv(df_projects, PKL_FILE_NAME)


if __name__ == "__main__":
    main()
