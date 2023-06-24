import pandas as pd
import requests
from sonarqube import SonarQubeClient
import json

url = 'http://localhost:9000'
username = "admin"
password = "admin21"

params = {
    'componentKeys': 'seatunnel-0066affacf7deff81cd685387fd5ab81241b8fa5',
    'languages': 'java',
    'types': 'CODE_SMELL',
    'ps': 500,
    'p': 1
}

sonar = SonarQubeClient(sonarqube_url=url, username=username, password=password)
response = sonar.projects.search_projects(param=params)

data_dumps = json.dumps(response)
data = json.loads(data_dumps)
