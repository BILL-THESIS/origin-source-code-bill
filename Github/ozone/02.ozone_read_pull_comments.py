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
    df2_issued_comments = pd.read_pickle("../output/ozone_filtered_issues_requests_comments.pkl")

    df1_all_pull = pd.read_pickle("../output/ozone_pulls.pkl")

    # filter the data not Nan in the required columns
    df2_issued_requests_not_nan = df2_issued_comments[df2_issued_comments['pull_request'].notna()]
    df2_issued_requests_not_nan['url_pull'] = df2_issued_requests_not_nan['pull_request'].apply(lambda x: x['url'])

    # get the data from the urls
    df3_pull_url = get_data_from_urls(df2_issued_requests_not_nan['url_pull'])
    df3_pull_url = pd.json_normalize(df3_pull_url)

    df4 = df3_pull_url[df3_pull_url['merged_at'].notna()]

    df4['merged_at'] = pd.to_datetime(df4['merged_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
    print("Data date time part Year :", df4['merged_at'].dt.year)
    print("Data date time part Month :", df4['merged_at'].dt.month)

    # check year and month of the data

    # Filter data where the year is 2024 or later and the month is May or later
    after_2024 = df4[(df4['merged_at'].dt.year >= 2024) & (df4['merged_at'].dt.month >= 5)]

    # Filter data where the year is before 2024
    # OR if in 2024, it is before May
    before_2024 = df4[(df4['merged_at'].dt.year < 2024) |
                          ((df4['merged_at'].dt.year == 2024) & (df4['merged_at'].dt.month < 3))]

    # # filter the data not Nan in the required columns
    # df4_list_pull_url_api_not_nan = df3_list_pull_url_api[df3_list_pull_url_api['commits_url'].notna()]
    # df3_merged_pull.to_pickle("../output/ozone_filtered_issues_requests_comments_pulls.pkl")
