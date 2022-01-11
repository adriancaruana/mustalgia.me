# Last.fm no longer returns covers for artists, so this uses the webpages to circumvent this
import logging
from typing import List, Tuple

import aiohttp
import asyncio
import bs4
import pandas as pd
import requests


from _utils import Album, Artist


COVER_CACHE = {}


FALLBACK_COVER = (
    "https://lastfm.freetls.fastly.net/i/u/300x300/2a96cbd8b46e442fc41c2b86b821562f.png"
)


def ua_headers(ua: str = None):
    if ua is None:
        ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    return {"User-Agent": ua}


async def fetch(artist: str, album: str, session):
    url = f"https://www.last.fm/music/{artist.replace(' ', '+')}"
    if album is not None:
        url = url + f"/{album.replace(' ', '+')}"

    async with session.get(url) as response:
        strainer = bs4.SoupStrainer(
            "div", attrs={"class": "header-new-background-image"}
        )

        html = await response.text()
        if response.status != 200:
            logging.warning(f"No cover found for: {url}")
            return FALLBACK_COVER
        soup = bs4.BeautifulSoup(html, "lxml", parse_only=strainer)
        try:
            return str(soup.find(class_="header-new-background-image").get("content"))
        except Exception as e:
            return FALLBACK_COVER


async def async_get_cover(artists: List[str], albums: List[str]):
    # headers = ua_headers()
    if albums is None:
        albums = [None for _ in artists]

    tasks = []

    async with aiohttp.ClientSession() as session:
        for artist, album in zip(artists, albums):
            task = asyncio.ensure_future(fetch(artist, album, session))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        return responses


def get_covers(artists: List[str], albums: List[str] = None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(async_get_cover(artists=artists, albums=albums))
    r = loop.run_until_complete(future)
    assert all(i.endswith(".jpg") or i.endswith(".png") for i in r)
    return r


def fill_df_with_covers(df: pd.DataFrame):
    global COVER_CACHE

    artists = df.artist_name.tolist()
    albums = [None for _ in artists]
    if "album_name" in df.columns:
        albums = df.album_name.tolist()

    args = list(set((artist, album) for artist, album in zip(artists, albums)))
    args = list(arg for arg in args if arg not in COVER_CACHE)
    covers = get_covers([arg[0] for arg in args], [arg[1] for arg in args])

    for key, cover in zip(args, covers):
        if key in COVER_CACHE:
            logging.warning("This should not happen!")
            continue
        COVER_CACHE[key] = cover

    # rewrite the 'cover_uri' col with the updated urls
    df["cover_uri"] = df.apply(
        lambda row: COVER_CACHE[
            (row.artist_name, row.album_name if "album_name" in row.index else None)
        ],
        axis=1,
    )
    return df


def get_avatar(user: str):
    url = f"https://www.last.fm/user/{user}"
    response = requests.get(url)

    if response.status_code != 200:
        logging.warning(f"No avatar found for: {url}")
        return None

    html = response.text
    soup = bs4.BeautifulSoup(html, "lxml")
    try:
        return str(soup.find(class_="avatar").findChildren("img" , recursive=False)[0].get("src"))
    except Exception as e:
        return None


if __name__ == "__main__":
    artist = "Spellling"
    album = "The Turning Wheel"
    r = get_covers([artist], [album])
    print(r)
    artist = "Spellling"
    album = None
    r = get_covers([artist], [album])
    print(r)
