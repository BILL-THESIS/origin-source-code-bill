import pandas as pd
import requests
from Authorization import git_token
from concurrent.futures import ThreadPoolExecutor

# Configuration for repository details
# List of repositories (owner/repo)
REPOSITORIES = [
    ('apache', 'seatunnel'),
    ('apache', 'ozone'),
    ('apache', 'pulsar'),
]

# Headers for authentication
HEADERS = {
    'Authorization': f'token {git_token.token}',
    'Accept': 'application/vnd.github.v3+json',
}


def get_pull_requests_with_comments(url, min_comments=1):
    """
    Fetch pull requests with a minimum number of comments.

    Parameters:
        url (str): The API URL for pull requests/issues.
        min_comments (int): The minimum number of comments a pull request must have.

    Returns:
        List[dict]: A list of pull request dictionaries that meet the criteria.
    """
    pull_requests = []
    page = 1

    while True:
        response = requests.get(url, headers=HEADERS, params={'state': 'closed', 'page': page, 'per_page': 100})
        response.raise_for_status()  # Ensure we catch any errors
        data = response.json()

        if not data:
            break

        # Filter pull requests based on comment count
        pull_requests.extend(pr for pr in data if pr.get('comments', 0) >= min_comments)
        page += 1

    return pull_requests


def filter_pull_requests_by_label(pull_requests, label):
    """
    Filter pull requests by a specific label.

    Parameters:
        pull_requests (List[dict]): The list of pull request dictionaries.
        label (str): The label to filter by.

    Returns:
        List[dict]: A filtered list of pull requests containing the label.
    """
    return [
        pr for pr in pull_requests
        if any(lbl['name'] == label for lbl in pr.get('labels', []))
    ]


def save_pull_requests_to_file(pull_requests, filename):
    """
    Save pull requests to a file in pickle format.

    Parameters:
        pull_requests (List[dict]): The list of pull requests.
        filename (str): The file path where data will be saved.
    """
    df = pd.DataFrame(pull_requests)
    df.to_pickle(filename)
    print(f'Saved {len(pull_requests)} pull requests to "{filename}"')


def process_repositories(repositories):
    for owner, repo in repositories:
        pulls_url = f'https://api.github.com/repos/{owner}/{repo}/issues'
        print(f'Processing repository: {owner}/{repo}')

        # Fetch and filter pull requests
        filtered_pull_requests = get_pull_requests_with_comments(pulls_url, min_comments=1)
        bug_pull_requests = filter_pull_requests_by_label(filtered_pull_requests, 'bug')

        # Save the filtered pull requests to a file
        filename = f"../output/{repo}_filtered_issues_requests_bugs.pkl"
        save_pull_requests_to_file(bug_pull_requests, filename)




if __name__ == '__main__':
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        data = executor.map(lambda repo: process_repositories(*repo), REPOSITORIES)
