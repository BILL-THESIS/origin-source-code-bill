import aiohttp
import asyncio
import pandas as pd
from Authorization import git_token

url = 'https://api.github.com/repos/apache/ozone/pulls?state=closed&per_page=300&page='


async def fetch(session, api, page):
    async with session.get(api + str(page), headers=git_token.header) as response:
        return await response.json()


async def fetch_all(api, fromPage, toPage):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, api, i) for i in range(fromPage, toPage + 1)]
        results = await asyncio.gather(*tasks)
        return results


def cell_api(api, fromPage=1, toPage=5):
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(fetch_all(api, fromPage, toPage))
    # Flatten the list of lists
    return [item for sublist in data for item in sublist]


data_list_api = cell_api(url, fromPage=1, toPage=150)
df = pd.json_normalize(data_list_api)
df.to_pickle('../output/ozone_pulls.pkl')
