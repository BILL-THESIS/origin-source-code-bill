import sonarqube
from sonarqube import SonarQubeClient

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

params = {
    'scopes': 'MAIN',
    'types': 'CODE_SMELL',
    'ps': 100,
    'p': 1  # Start with the first page
}

all_issues = []  # List to store all the issues

while True:
    response = sonar.issues.search_issues(**params)
    issues = response['issues']

    all_issues.extend(issues)

    if 'paging' in response:
        paging = response['paging']
        total = paging['total']
        page_index = paging['pageIndex']
        page_size = paging['pageSize']

        if page_index * page_size >= total:
            break  # Break the loop if we have reached the last page

        params['p'] += 1  # Move to the next page

# Now you have all the issues in the `all_issues` list
# You can process the issues as desired
for issue in all_issues:
    issue_key = issue['key']
    issue_type = issue['type']
    # ... process other issue properties ...

    print(f"Issue Key: {issue_key}, Issue Type: {issue_type}")
