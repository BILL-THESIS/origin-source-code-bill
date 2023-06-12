import pandas as pd

# Sample JSON data
data = [
    {
        "state": "Florida",
        "counties": [
            {"name": "Dade", "population": 12345},
            {"name": "Broward", "population": 40000},
            {"name": "Palm Beach", "population": 60000},
        ],
    },
    {
        "state": "Ohio",
        "counties": [
            {"name": "Summit", "population": 1234},
            {"name": "Cuyahoga", "population": 1337},
        ],
    },
],
[
    {
        "state": "bill",
        "counties": [
            {"name": "de", "population": 1245},
            {"name": "Bro", "population": 4000},
            {"name": "Beach", "population": 600},
        ],
    },
    {
        "state": "Ori",
        "counties": [
            {"name": "miter", "population": 8934},
            {"name": "hogayu", "population": 3430},
        ],
    },
]

# # Normalize JSON data into a flat table
# df = pd.json_normalize(data, 'counties', ['state'])
#
# # Print the flattened table
# print(df)

df = pd.DataFrame()

for sublist in data:
    for dictionary in sublist:
        flattened_data = pd.json_normalize(dictionary, 'counties', ['state'])
        df = df.append(flattened_data, ignore_index=True)

print(df)