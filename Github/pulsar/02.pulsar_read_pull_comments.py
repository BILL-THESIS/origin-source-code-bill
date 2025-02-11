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
        results = await asyncio.gather(*tasks)
        return results


def get_data_from_urls(df: pd.DataFrame) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    urls = df.tolist()
    data_list_url = loop.run_until_complete(fetch_all(urls))
    return data_list_url


if __name__ == '__main__':
    # Load the filtered pull requests
    df_issued_comments = pd.read_pickle("../output/pulsar_filtered_issues_requests_comments.pkl")

    # filter the data not Nan in the required columns
    df_issued_requests_not_nan = df_issued_comments[df_issued_comments['pull_request'].notna()]
    df_issued_requests_not_nan['url_pull'] = df_issued_requests_not_nan['pull_request'].apply(lambda x: x['url'])

    # get the data from the urls
    data_pull = get_data_from_urls(df_issued_requests_not_nan['url_pull'])
    data_pull = pd.json_normalize(data_pull)

    # filter the data not Nan in the required columns
    data_list_pull_url_api_not_nan = data_pull[data_pull['merged_at'].notna()]
    data_list_pull_url_api_not_nan.to_pickle("../output/pulsar_filtered_issues_requests_comments_pulls_new.pkl")
