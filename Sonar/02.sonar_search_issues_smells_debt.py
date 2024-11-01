import pandas as pd
from sonarqube import SonarQubeClient
from Authorization import sonar_token

# Constants
PAGE_SIZE = 500
# svae file name
PKL_FILE_NAME = "../Sonar/output/02.sonar_debt.pkl"
# read project file name
PROJECTS_PKL = "../Sonar/output/01.sonar_search_pull.pkl"


def get_sonar_client(url, username, password):
    """Initialize and return a SonarQubeClient."""
    return SonarQubeClient(sonarqube_url=url, username=username, password=password)


def fetch_issues_for_project(sonar_client, project_key, page_size):
    """Fetch all issues for a given project from SonarQube API."""
    params = {
        'componentKeys': project_key,
        'scopes': 'MAIN',
        'languages': 'java',
        'types': 'CODE_SMELL',
        'ps': page_size,
        'p': 1
    }

    all_issues = []

    while True:
        response = sonar_client.issues.search_issues(**params)
        issues = response.get('issues', [])
        if not issues:
            break

        all_issues.extend(issues)

        if len(issues) < page_size:
            break

        params['p'] += 1

    return all_issues


def process_issues(issues):
    """Process issues and extract required data."""
    data_list = []

    for issue in issues:
        data = {
            'key': issue['key'],
            'type': issue['type'],
            'rule': issue['rule'],
            'project': issue['project'],
            'effort': issue.get('effort', ''),
            'debt': issue.get('debt', '')
        }
        data_list.append(data)

    return data_list


def group_and_pivot_issues(data_list):
    """Group and pivot issues data."""
    df = pd.DataFrame(data_list).dropna(axis='columns')
    df.set_index(['project', 'rule'], inplace=True)
    rule_counts = df.groupby(['project', 'rule']).size().reset_index(name='counts')
    pivot_df = rule_counts.pivot_table(index='project', columns='rule', values='counts', fill_value=0)
    return pivot_df


def save_to_pickle(df, file_name):
    df.to_pickle(file_name)
    print(f"Data successfully exported to {file_name}")


def main():
    sonar_client = get_sonar_client(sonar_token.URL, sonar_token.USERNAME, sonar_token.PASSWORD)
    project_keys = pd.read_pickle(PROJECTS_PKL)['key']

    all_issues = []
    for project_key in project_keys:
        print(f"Processing project: {project_key}")
        issues = fetch_issues_for_project(sonar_client, project_key, PAGE_SIZE)
        all_issues.extend(process_issues(issues))
    pivot_df = group_and_pivot_issues(all_issues)
    save_to_pickle(pivot_df, PKL_FILE_NAME)


if __name__ == "__main__":
    main()
