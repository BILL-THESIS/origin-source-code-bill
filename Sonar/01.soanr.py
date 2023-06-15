from sonarqube import SonarQubeClient
import pandas as pd
# SonarQubeClient

url = 'http://localhost:9000'
username = "admin"
password = "admin21"
sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)

projects = sonar.projects.search_projects(p=50)

df = pd.DataFrame.from_dict(projects['components'])