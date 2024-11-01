import pandas as pd
from sonarqube import SonarQubeClient
from Authorization import sonar_token

# Constants
PAGE_SIZE = 500
# svae file name
PKL_FILE_NAME = "../Sonar/output/02.sonar_measures_smells.pkl"
# read project file name
PROJECTS_PKL = "../Sonar/output/01.sonar_search_pull.pkl"


def get_sonar_client(url, username, password):
    """Initialize and return a SonarQubeClient."""
    return SonarQubeClient(sonarqube_url=url, username=username, password=password)


if __name__ == "__main__":
    # Initialize the SonarQube client
    sonar_client = get_sonar_client(sonar_token.URL, sonar_token.USERNAME, sonar_token.PASSWORD)

    # Read the project keys from the pickle file
    df_projects = pd.read_pickle(PROJECTS_PKL)

    # Fetch measures for each project
    all_measures = []
    for project_key in df_projects['key']:
        measures = sonar_client.measures.get_component_with_specified_measures(component=project_key,
                                                                 fields="metrics",
                                                                 metricKeys="code_smells")
        all_measures.append(measures)
        print(f"Measures for project {project_key} fetched.")
    # Convert to DataFrame and save
    df = pd.json_normalize(all_measures)

    # separate the measures column component.measures from the dictionary
    df_measures = pd.json_normalize(df['component.measures'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}))
    df_concat = pd.concat([df, df_measures], axis=1)
    df_concat.to_pickle(PKL_FILE_NAME)
    print(f"Data saved to {PKL_FILE_NAME} Done.")
    print(df_concat.head())
