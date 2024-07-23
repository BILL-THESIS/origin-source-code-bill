import json
import pandas as pd
from sonarqube import SonarQubeClient
from Authorization import sonar_token


def create_sonar_client(url, username, password):
    """Create and return a SonarQube client."""
    return SonarQubeClient(sonarqube_url=url, username=username, password=password)


def fetch_sonar_rules(client, params):
    """Fetch SonarQube rules with the given parameters."""
    try:
        return client.rules.search_rules(**params)
    except Exception as e:
        print(f"Error fetching rules: {e}")
        return None


def get_rules_data(client, language='java', types='CODE_SMELL', page_size=500):
    """Retrieve all rules data, handling pagination."""
    params = {
        'languages': language,
        'types': types,
        'ps': page_size,
        'p': 1
    }
    all_rules = []

    while True:
        data = fetch_sonar_rules(client, params)
        if not data or 'rules' not in data:
            break

        all_rules.extend(data['rules'])

        # Check if there are more pages
        if 'paging' in data and 'pageIndex' in data['paging'] and data['paging']['pageIndex'] < data['paging']['total']:
            params['p'] += 1
            print(f"Loading page {params['p']}...")
        else:
            break

    return all_rules


def main():
    # Initialize the SonarQube client
    sonar_client = create_sonar_client(sonar_token.URL, sonar_token.USERNAME, sonar_token.PASSWORD)

    # Fetch all Java CODE_SMELL rules
    rules_data = get_rules_data(sonar_client)

    # Normalize and convert to DataFrame
    if rules_data:
        df = pd.json_normalize(rules_data)
        df.to_pickle("../Sonar/output/sonar_rules_version9.9.6.pkl")
        print(df.head())
    else:
        print("No rules data fetched.")


if __name__ == "__main__":
    main()
