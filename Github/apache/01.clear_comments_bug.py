import pandas as pd
import requests
from Authorization import git_token
import time

# List of repositories (owner/repo)
REPOSITORIES = [
    ('apache', 'seatunnel'),
    ('apache', 'ozone'),
    ('apache', 'pulsar')
    # Add more repositories as needed
]

# Headers for authentication
HEADERS = {
    'Authorization': f'token {git_token.token}',
    'Accept': 'application/vnd.github.v3+json',
}


def get_pull_requests_with_comments(url, min_comments=1):
    """
    Fetch pull requests with a minimum number of comments.
    """
    pull_requests = []
    page = 1

    while True:
        response = requests.get(url, headers=HEADERS, params={'state': 'closed', 'page': page, 'per_page': 100})
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        pull_requests.extend(pr for pr in data if pr.get('comments', 0) >= min_comments)
        page += 1

    return pull_requests


def filter_pull_requests_by_labels(pull_requests, labels):
    """
    Filter pull requests by a list of labels.
    """
    return [
        pr for pr in pull_requests
        if any(lbl['name'] in labels for lbl in pr.get('labels', []))
    ]


def save_pull_requests_to_file(pull_requests, filename):
    """
    Save pull requests to a file in pickle format.
    """
    df = pd.DataFrame(pull_requests)
    df.to_pickle(filename)
    print(f'Saved {len(pull_requests)} pull requests to "{filename}"')


def process_repositories_sequentially(repositories):
    for owner, repo in repositories:
        try:
            pulls_url = f'https://api.github.com/repos/{owner}/{repo}/issues'
            print(f'Processing repository: {owner}/{repo}')

            # Fetch and filter pull requests
            filtered_pull_requests = get_pull_requests_with_comments(pulls_url, min_comments=1)
            bug_pull_requests = filter_pull_requests_by_labels(filtered_pull_requests, ['bug', 'type/bug'])

            # Save the filtered pull requests to a file
            filename = f"../output/{repo}_filtered_issues_requests_comments.pkl"
            save_pull_requests_to_file(filtered_pull_requests, filename)

            # Optional: Save bug_pull_requests separately if needed
            bug_filename = f"../output/{repo}_bug_pull_requests.pkl"
            save_pull_requests_to_file(bug_pull_requests, bug_filename)

            print(f'Completed processing for repository: {repo}. Proceeding to the next one.')

            # Optional delay to avoid hitting rate limits
            time.sleep(2)  # Adjust the sleep time as needed

        except requests.exceptions.RequestException as e:
            print(f'Error processing repository {repo}: {e}')
            continue


if __name__ == '__main__':
    # Process repositories one by one
    repo = process_repositories_sequentially(REPOSITORIES)
