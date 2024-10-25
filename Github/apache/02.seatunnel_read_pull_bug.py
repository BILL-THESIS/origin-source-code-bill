import requests
import aiohttp
import asyncio
import pandas as pd
from Authorization import git_token

async def fetch(session, url):
    async with session.get(url, headers=git_token.header) as response:
        return await response.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def get_data_from_urls(urls):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(fetch_all(urls))

def main():
    pull_bug_files = {
        'pulsar': '../output/pulsar_bug_pull_requests.pkl',
        'ozone': '../output/ozone_bug_pull_requests.pkl',
        'seatunnel': '../output/seatunnel_bug_pull_requests.pkl'
    }

    for repo, file in pull_bug_files.items():
        df = pd.read_pickle(file)
        df_not_nan = df[df['pull_request'].notna()]
        df_not_nan['url_pull'] = df_not_nan['pull_request'].apply(lambda x: x['url'])

        data_list_pull_url_api = get_data_from_urls(df_not_nan['url_pull'].tolist())
        data_list_pull_url_api = pd.json_normalize(data_list_pull_url_api)

        data_list_pull_url_api_not_nan = data_list_pull_url_api[data_list_pull_url_api['merged_at'].notna()]
        output_file = f"../output/{repo}_filtered_issue_bug.pkl"
        data_list_pull_url_api_not_nan.to_pickle(output_file)

# if __name__ == '__main__':
#     main()