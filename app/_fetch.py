# Script to fetch Last.fm listening history
# Author: Adrian caruana (adrian@adriancaruana.com)
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
from datetime import datetime
import logging
import time
from pathlib import Path
from typing import Tuple, Dict
import shutil
import sys
import xml

import pandas as pd
import pylast
from tqdm import tqdm

from _cache import Cache
from _utils import (
    Album,
    Artist,
    Track,
    _number,
    cleanup_nodes,
    _extract,
    _get_cover,
    _get_mbid,
)
from _covers import get_avatar

N_REQUEST_THREADS = 4
RETRY_SLEEP = 3  # seconds
FETCH_LIMIT_LOW = 50  # scrobbles/page
FETCH_LIMIT_HIGH = 1000  # scrobbles/page
TRACKS_IN_MEMORY_LIMIT = 1000
MAX_TOTAL_SCROBBLES = 1_000_000


@dataclasses.dataclass
class Fetch:
    pkey: str
    skey: str
    user: str
    cache: Cache = None
    _network: pylast.LastFMNetwork = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        self._network = pylast.LastFMNetwork(
            api_key=self.pkey,
            api_secret=self.skey,
            username=self.user,
            password_hash=None,
        )
        if self._get_total_scrobbles() > MAX_TOTAL_SCROBBLES:
            raise ValueError(
                f"Whoa, you've got over 1M scrobbles! Unfortunately, I can't let you run this user on my server, since we must be nice to the Last.fm API. Instead, send me an email and I'll run it for you manually."
            )
        if self.cache is None:
            self.cache = Cache(self.user)
        self.total_pages = None

    def _node_to_track(self, node):
        artist = Artist(
            artist_name=_extract(node.getElementsByTagName("artist")[0], "name"),
            cover_uri=_get_cover(node, artist=True),
            mbid=_get_mbid(node, artist=True),
        )
        album = Album(
            album_name=_extract(node, "album"),
            artist=artist,
            cover_uri=_get_cover(node),
            mbid=_get_mbid(node),
        )
        return Track(
            track_name=_extract(node, "name", index=1),
            time_stamp=int(node.getElementsByTagName("date")[0].getAttribute("uts")),
            date_played=_extract(node, "date"),
            loved=bool(int(_extract(node, "loved"))),
            album=album,
            artist=artist,
        )

    def _get_total_scrobbles(self):
        user = pylast.User(self.user, self._network)
        method_name = user.ws_prefix + ".getInfo"
        params = dict(
            user=self.user,  # user (Required) : The user to fetch info for. Defaults to the authenticated user.
            api_key=self.pkey,  # api_key (Required) : A Last.fm API key.
        )
        node = cleanup_nodes(user._request(method_name, False, params))
        playcount = int(_extract(node, "playcount"))
        return playcount

    def _fast_fetch_tracks(self, limit=FETCH_LIMIT_LOW):
        user = pylast.User(self.user, self._network)

        def _fetch_page(page: int, retry_counter: int = 0):
            method_name = user.ws_prefix + ".getRecentTracks"
            params = dict(
                # Limit is actually 1k, undocumented... :)
                limit=limit,  # limit (Optional) : The number of results to fetch per page. Defaults to 50. Maximum is 200.
                user=self.user,  # user (Required) : The last.fm username to fetch the recent tracks of.
                page=page,  # page (Optional) : The page number to fetch. Defaults to first page.
                extended=1,  # extended (0|1) (Optional) : Includes extended data in each artist, and whether or not the user has loved each track
                api_key=self.pkey,  # api_key (Required) : A Last.fm API key.
            )
            try:
                return cleanup_nodes(user._request(method_name, False, params))
            except Exception as e:
                print(str(e))
                if str(e) == "User not found":
                    raise e
                if retry_counter == 3:
                    raise RuntimeError(
                        f"Maximum retrys exceeded. Request failed: {str(e)}"
                    )
                logging.warning(
                    f"Unexpected error occurred: {str(e)}. Retrying in 30s."
                )
                time.sleep(30)
                return _fetch_page(page=page, retry_counter=retry_counter + 1)

        # Fetch only the first page to begin, since we don't know the total number of pages
        doc = _fetch_page(1)  # This is blocking
        # break if there are no child nodes
        if not doc.documentElement.childNodes:
            raise ValueError("No child nodes in first page!")
        main = doc.documentElement.childNodes[0]
        if main.hasAttribute("totalPages") or main.hasAttribute("totalpages"):
            total_pages = _number(
                main.getAttribute("totalPages") or main.getAttribute("totalpages")
            )
            self.total_pages = total_pages
        else:
            raise RuntimeError("No total pages attribute")

        for node in main.childNodes:
            if not node.nodeType == xml.dom.Node.TEXT_NODE:
                if node.getAttribute("nowplaying") == "true":
                    continue  # Not interested in considering the currently playing track
                yield self._node_to_track(node)._to_row()

        futures = {}
        with ThreadPoolExecutor(max_workers=N_REQUEST_THREADS) as ex:
            # Generator should be ordered, so use dictionary to ensure order of futures.
            for page in range(2, 2 + N_REQUEST_THREADS):
                futures[page] = ex.submit(_fetch_page, page=page)  # back pressure
            for page in tqdm(range(2, total_pages + 1), ncols=70):
                if page <= total_pages + 1:
                    futures[page + N_REQUEST_THREADS] = ex.submit(
                        _fetch_page, page=page + N_REQUEST_THREADS
                    )  # trickle
                doc = futures[page].result()  # This is blocking
                # break if there are no child nodes
                if not doc.documentElement.childNodes:
                    break
                main = doc.documentElement.childNodes[0]
                for node in main.childNodes:
                    if not node.nodeType == xml.dom.Node.TEXT_NODE:
                        if node.getAttribute("nowplaying") == "true":
                            continue  # Not interested in considering the currently playing track
                        yield self._node_to_track(node)._to_row()
                del futures[page]  # html can be large, so good idea to free memory

    def _write_cache(self, df: pd.DataFrame=None, complete=True):
        time = int(datetime.now().timestamp())
        meta = {
            "timestamp": time,
            "user": self.user,
            "complete": complete,
            "total_pages": self.total_pages,
        }
        if df is not None:
            self.cache.write_scrobbles_part(meta=meta, df=df)
        else:
            self.cache.write_json("meta.json", meta)

    def _full_fetch(self):
        # Fetch it all
        logging.warning("Cache doesn't exist. Running full fetch and caching")
        in_memory = []
        # Write meta to say that the fetching is incomplete to begin with
        self._write_cache(complete=False)
        # Use high limit (more tracks per page) to reduce API calls
        for track in self._fast_fetch_tracks(limit=FETCH_LIMIT_HIGH):
            in_memory.append(track)
            if len(in_memory) > TRACKS_IN_MEMORY_LIMIT:
                df = pd.DataFrame(in_memory)
                self._write_cache(df, complete=False)
                in_memory = []

        df = pd.DataFrame(in_memory)
        self._write_cache(df, complete=True)
        meta, df = self.cache.read_scrobbles_parquet()
        return meta, df

    def fetch_and_cache(self) -> Tuple[Dict, pd.DataFrame]:
        if not self.cache.scrobbles_path.exists():
            return self._full_fetch()
        if (
            self.cache.meta_path.exists()
            and self.cache.read_meta()["complete"] is False
        ):
            logging.warning("Cache exists, but is incomplete. Remove and refetch.")
            shutil.rmtree(self.cache.scrobbles_path)
            return self._full_fetch()

        # Continue to fetch until caught up to cache
        meta, cached_df = self.cache.read_scrobbles_parquet()
        most_recent_cached_row = cached_df[
            cached_df.time_stamp == cached_df.time_stamp.max()
        ].iloc[0]
        uncached_tracks = []
        for track in iter(self._fast_fetch_tracks()):
            if (
                track["track_name"] == most_recent_cached_row["track_name"]
                and track["time_stamp"] == most_recent_cached_row["time_stamp"]
            ):
                break
            uncached_tracks.append(track)

        if len(uncached_tracks) == 0:
            logging.warning("No tracks have been scrobbled since last cache.")
            return meta, cached_df

        logging.warning(
            f"Found {len(uncached_tracks)} uncached tracks, prepending them to the cache."
        )
        self._write_cache(df=pd.DataFrame(uncached_tracks), complete=True)
        updated_meta, updated_df = self.cache.read_scrobbles_parquet()
        return updated_meta, updated_df

    def _get_avatar(self):
        if (self.cache._cache_dir / "avatar.json").exists():
            return self.cache.read_json("avatar.json")["url"]
        data = {"url": get_avatar(self.user)}
        self.cache.write_json("avatar.json", data)
        return data["url"]


if __name__ == "__main__":
    f = Fetch(
        pkey=sys.argv[1],
        skey=sys.argv[2],
        user=sys.argv[3],
    )
    f.fetch_and_cache()
