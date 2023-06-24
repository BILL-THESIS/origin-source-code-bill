import pandas as pd
import requests
from Authorization import git_token

df = pd.read_csv("../output/ozone.csv")
data_list = []

for i in df['url']:
    data = requests.get(i, headers=git_token.header)
    data_list.append(data.json())
    print("========= :" , data_list)
    obj = pd.json_normalize(data_list)
    print(obj)

df3 = pd.DataFrame(obj)
print(df)
df3.to_csv("ozone_check_pull_1.csv" , index=False)