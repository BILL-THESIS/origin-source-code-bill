import pandas as pd
import requests
from Authorization import git_token

# Replace with your own token and repository details
owner = 'apache'
repo = 'pulsar'

# Headers for authentication
headers = {
    'Authorization': f'token {git_token.token}',
    'Accept': 'application/vnd.github.v3+json',
}


# Function to get pull requests with more than one conversation
def get_pull_requests_with_comments(url):
    pull_requests = []
    page = 1

    while True:
        response = requests.get(url, headers=headers, params={'state': 'closed', 'page': page, 'per_page': 100})
        data = response.json()

        if not data:
            break

        for pr in data:
            if pr['comments'] >= 1:
                pull_requests.append(pr)

        page += 1

    return pull_requests


# Pull Requests URL
pulls_url = f'https://api.github.com/repos/{owner}/{repo}/issues'

# Fetch pull requests with more than one comment
filtered_pull_requests = get_pull_requests_with_comments(pulls_url)

# Save the filtered pull requests to a file
df = pd.DataFrame(filtered_pull_requests)
df.to_pickle("../output/pulsar_filtered_issues_requests_comments.pkl")
print(
    f'Saved {len(filtered_pull_requests)} pull requests with more than one conversation to "filtered_pull_requests.pickle"')
