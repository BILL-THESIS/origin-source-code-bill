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


def get_data_from_urls(df):
    loop = asyncio.get_event_loop()
    urls = df['url'].tolist()
    data_list_url = loop.run_until_complete(fetch_all(urls))
    return data_list_url



df_ozone_pull_closed = pd.read_pickle("../output/ozone_pulls.pkl")
data_list_pull_usl_api = get_data_from_urls(df_ozone_pull_closed)
df_pull_url_api = pd.json_normalize(data_list_pull_usl_api)
df_pull_url_api.to_pickle("ozone_pull_url_api.pkl")

