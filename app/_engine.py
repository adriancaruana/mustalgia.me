# Last.fm listening history insight engine
# Author: Adrian caruana (adrian@adriancaruana.com)
import dataclasses
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, Union, List, Tuple, Optional
import pytz
import os

from tqdm import tqdm
from ripser import ripser
import pandas as pd
import numpy as np

from _utils import Track, Album, Artist, DUMMY_ALBUM, DUMMY_ARTIST
from _cache import Cache
from _covers import fill_df_with_covers


# LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo
# TODAY = datetime.now(pytz.utc)
TODAY = datetime.now()
DELTA_YEAR = timedelta(days=365)
DELTA_MONTH = timedelta(days=(365 // 12))
DELTA_FNIGHT = timedelta(days=14)
DELTA_WEEK = timedelta(days=7)
DELTA_DAY = timedelta(days=1)
RIPSER_N_PERM_FACTOR = 4
HISTORICAL_REFRESH_FREQ = DELTA_MONTH
FORCE_FORGOTTEN_FAST = os.getenv("FORCE_FORGOTTEN_FAST", False)
FORGOTTEN_FAST_SCROBBLE_THRESH = 200_000
FORGOTTEN_FAST = os.getenv("FORGOTTEN_FAST", False)
HISTORICAL_META = "historical_meta.json"
HISTORICAL_ARTEFACTS_TYPES = [
    "favourites_data",
    "obsessions_data",
    "infatuations_data",
    "forgotten_data",
    "on_repeat_data",
]
HISTORICAL_ARTEFACTS_KEYS = [
    f"{h}_{t}" for h in HISTORICAL_ARTEFACTS_TYPES for t in ["album", "artist"]
]
HISTORICAL_TITLES_AND_DESCRIPTIONS = [
    (
        "Favourites",
        "<em>Favourites</em> refer to music that this user listens to both <b>often</b> and <b>consistently</b> (since the first scrobble).",
    ),
    (
        "Infatuations",
        "<em>Infatuations</em> refer to music that the user has recently been listenting to a lot.",
    ),
    (
        "Flings",
        "<em>Flings</em> refers to music that the user listened to a lot, but not recently. That is, <em>flings</em> are past-<em>infatuations</em>.",
    ),
    (
        "Forgotten",
        "<em>Forgotten</em> refers to music that the user listened to only a couple of times, but never again.",
    ),
    (
        "On Repeat",
        "<em>On Repeat</em> refers to music that the user listened to, at some point, most exclusively.",
    ),
]
TIMELY_ARTEFACTS_KEYS = ["on_this_x_data"]
TIMELY_TITLES_AND_DESCTIPTIONS = [
    (
        "On this Day",
        "<em>On this Day</em> refers to music that the user listened to on this day in previous years.",
    )
]


@dataclasses.dataclass
class Config:
    # Timeseries
    width: Optional[timedelta] = DELTA_MONTH
    stride: Optional[timedelta] = DELTA_FNIGHT
    # On this x
    interval: Optional[timedelta] = DELTA_YEAR
    window: Optional[timedelta] = 2 * DELTA_DAY
    on_this_x_min_scrobbles: Optional[int] = 4
    # Favourites
    max_results_favourites: Optional[int] = 20
    # Infatuations & Obsessions
    infatuation_period: Optional[timedelta] = 2 * DELTA_MONTH
    infatuation_z_score: Optional[float] = 1.5
    max_results_infatuations: Optional[int] = 100
    obsession_z_score: Optional[float] = 0
    # On repeat
    min_streak_length: Optional[int] = 10
    max_results_repeat: Optional[int] = 20
    # Forgotten
    forget_period: Optional[timedelta] = 6 * DELTA_MONTH
    persistence_interval: Optional[tuple] = (3, 15)
    n_session_thresh: Optional[int] = 3
    max_results_forgotten: Optional[int] = 100


@dataclasses.dataclass
class Engine:
    df: pd.DataFrame
    config: Config = None
    cache: Cache = None
    scrobbles: List[Track] = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if self.config is None:
            self.config = Config()

    def generate_historical_artefacts(self):
        # Init
        if self.scrobbles is None:
            self.scrobbles = [
                Track._from_row(row) for _, row in tqdm(self.df.iterrows())
            ]
        self.tracks = set(self.scrobbles)
        self.albums = set(t.album for t in self.tracks)
        self.artists = set(t.artist for t in self.tracks)
        # Start analysing data
        self.features, self.bin_centres = self._convolve(self.df)
        self.timeseries = {
            "all": self._get_features(self.features, join_on=None),
            "album": self._get_features(self.features, join_on=Album),
            "artist": self._get_features(self.features, join_on=Artist),
        }
        self.obsessions = None
        self.infatuations = None
        # Run: Data
        _historical_data = {
            historical_key: getattr(self, historical_key)()
            for historical_key in HISTORICAL_ARTEFACTS_TYPES
        }
        # Flatten the data structure, and add album/artist covers post-hoc
        historical_data = {
            f"{k}_{t}": fill_df_with_covers(_historical_data[k][t])
            # f"{k}_{t}": _historical_data[k][t]
            for k in _historical_data.keys()
            for t in ["album", "artist"]
        }
        assert len(historical_data) == len(HISTORICAL_ARTEFACTS_KEYS)
        for k, v in historical_data.items():
            self.cache.write_df(k, v)
        # Run: Meta
        historical_meta = {
            "last_updated": self._dt_to_int(TODAY),
        }
        self.cache.write_json(HISTORICAL_META, historical_meta)
        return historical_meta, historical_data

    def generate_timely_artefacts(self):
        timely_artefacts = {
            timely_key: getattr(self, timely_key)()
            for timely_key in TIMELY_ARTEFACTS_KEYS
        }
        return timely_artefacts

    def run_engine(self):
        # Always generate timely artefacts on-demand
        timely_artefacts = self.generate_timely_artefacts()
        # Generate historical artefacts if they haven't been done in a while, or ever
        if not self.cache.key_to_path("historical_meta.json").exists():
            _, historical_artefacts = self.generate_historical_artefacts()
            return timely_artefacts, historical_artefacts
        last_updated = self.cache.read_json("historical_meta.json")["last_updated"]
        if self._ts_to_dt(last_updated) + HISTORICAL_REFRESH_FREQ > TODAY:
            # Read historical artefacts
            historical_artefacts = {}
            for historical_key in HISTORICAL_ARTEFACTS_KEYS:
                historical_artefacts[historical_key] = self.cache.read_df(
                    historical_key
                )
            return timely_artefacts, historical_artefacts
        _, historical_artefacts = self.generate_historical_artefacts()
        return timely_artefacts, historical_artefacts

    @classmethod
    def _ts_to_dt(cls, timestamp: int) -> datetime:
        """Convert a timestamp to a datetime object"""
        return datetime.utcfromtimestamp(timestamp)

    @classmethod
    def _ts_to_year(cls, timestamp: int) -> datetime:
        return cls._ts_to_dt(timestamp).strftime("%Y")

    @classmethod
    def _dt_to_int(cls, t: Union[datetime, timedelta]) -> int:
        if isinstance(t, timedelta):
            return int(t.total_seconds())
        return int(t.timestamp())

    def _convolve(
        self,
        scrobbles: pd.DataFrame,
    ) -> Dict[int, pd.Int64Index]:
        width = self.config.width
        stride = self.config.stride
        # If performance becomes an issue, increase stride
        start = scrobbles.time_stamp.min()
        end = scrobbles.time_stamp.max()
        bin_centres = list(range(start, end, self._dt_to_int(stride)))
        unix_width = self._dt_to_int(width)
        features = {
            bin_centre: scrobbles[
                np.abs(scrobbles.time_stamp - bin_centre) <= (unix_width / 2)
            ].index
            for bin_centre in bin_centres
        }
        return features, bin_centres

    @classmethod
    def _windows_by_year(cls, window: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # This might crossover start/end of year, but that's only a problem .2% of the time.
        years = pd.Series(window.time_stamp).apply(cls._ts_to_year)
        return {year: window[years == year] for year in years.unique()}

    def _get_window(self) -> pd.DataFrame:
        interval = self._dt_to_int(self.config.interval)
        window = self._dt_to_int(self.config.window)
        offset = self._dt_to_int(TODAY) % interval
        deltas = (self.df.time_stamp.to_numpy() - offset) % interval
        mask = (deltas <= window) | deltas >= (interval - window)
        return self.df.iloc[mask]

    def _get_features(
        self,
        features: Dict[int, pd.Int64Index],
        join_on: Union[Album, Artist, None] = None,
    ):
        if join_on is None:
            return [len(indices) for indices in features.values()]

        if not (join_on is Album or join_on is Artist):
            raise ValueError(f"Unrecognised type to join on: {join_on=}.")

        # Messy but efficient
        attr = "album" if join_on is Album else "artist"
        timestamp_list = []
        for indices in tqdm(features.values()):
            timestamp_features = [self.scrobbles[idx] for idx in indices]
            timestamp_dict = {}
            for f in timestamp_features:
                if getattr(f, attr) not in timestamp_dict:
                    timestamp_dict[getattr(f, attr)] = 1
                else:
                    timestamp_dict[getattr(f, attr)] += 1
            timestamp_list.append(timestamp_dict)

        return {
            obj: [timestamp_dict.get(obj, 0) for timestamp_dict in timestamp_list]
            for obj in tqdm(getattr(self, attr + "s"))
        }

    def on_this_x_data(self):
        """Return 'On this [day]'-like recommendations."""
        windows_by_year = self._windows_by_year(self._get_window())

        on_this_x = {}
        for year, df in windows_by_year.items():
            if self._ts_to_year(self._dt_to_int(TODAY)) == year:
                continue
            # Init
            tracks = [Track._from_row(row) for _, row in df.iterrows()]
            albums = {}
            artists = {}
            for t in tracks:
                albums[t.album] = albums.get(t.album, 0) + 1
                artists[t.artist] = artists.get(t.artist, 0) + 1

            albums = pd.DataFrame(
                [{**album._to_row(), "value": count} for album, count in albums.items()]
            )
            albums = albums[
                albums["value"] >= self.config.on_this_x_min_scrobbles
            ].sort_values(["value"], ascending=False)
            artists = pd.DataFrame(
                [
                    {**artist._to_row(), "value": count}
                    for artist, count in artists.items()
                ]
            )
            artists = artists[
                artists["value"] >= self.config.on_this_x_min_scrobbles
            ].sort_values(["value"], ascending=False)
            on_this_x[year] = {
                "album": fill_df_with_covers(albums),
                "artist": fill_df_with_covers(artists),
            }
            # on_this_x[year] = {"album": albums, "artist": artists}

        return on_this_x

    def favourites_data(self):
        """Return 'Favourites' recommendations.
        This indicates sessions where a single album/artist
        was listened to almost exclusively.
        """
        all = np.asarray(self.timeseries["all"])

        def _get_favourites(attr):
            attr_timeseries = {
                obj: np.asarray(features) / all
                for obj, features, in self.timeseries[attr].items()
            }
            df = pd.DataFrame(
                [
                    {
                        **obj._to_row(),
                        "value": np.mean(timeseries[~np.isnan(timeseries)]),
                    }
                    for obj, timeseries, in attr_timeseries.items()
                ]
            ).sort_values(["value"], ascending=False)
            return df.iloc[: min(self.config.max_results_favourites, len(df))]

        return {
            "album": _get_favourites("album"),
            "artist": _get_favourites("artist"),
        }

    def infatuations_and_obsessions(self):
        """Return 'Infatuations' recommendations.
        This indicates albums/artists that were listened
        to a lot but during a brief period of time.
        Obsessions current/recent infatuations.
        Music with listening bursts that are greater than
        INFATUATION_Z_SCORE are counted as infatuations/obsessions.
        """
        # Infatuations are now called Flings
        # And Obsessions are now called Infatuations

        def _get(attr: str):
            period = self.config.infatuation_period
            non_zero_mask = np.asarray(self.timeseries["all"]) != 0
            di = {}
            timestamp = {}
            for obj, timeseries in self.timeseries[attr].items():
                # Fill zero with nan to preserve index
                non_zero_timeseries = np.where(
                    non_zero_mask, np.asarray(timeseries), np.nan
                )
                # mean = np.nanmean(non_zero_timeseries)
                # std = np.nanstd(non_zero_timeseries)
                # non_zero_timeseries = (non_zero_timeseries - mean) / std
                di[obj] = np.nanmax(non_zero_timeseries[~np.isnan(non_zero_timeseries)])
                timestamp[obj] = self.bin_centres[np.nanargmax(non_zero_timeseries)]

            _df = pd.DataFrame(
                [
                    {**obj._to_row(), "value": value, "time_stamp": time_stamp}
                    for (obj, value), time_stamp in zip(di.items(), timestamp.values())
                ]
            ).sort_values(["value"], ascending=False)

            def filter_df(_df, thresh):
                mean, std = _df["value"].mean(), _df["value"].std()
                _df["value"] = (_df["value"] - mean) / std
                return _df[_df["value"] >= thresh]

            split = self._dt_to_int(TODAY - period)

            infatuations = _df[_df.time_stamp < split]
            obsessions = _df[_df.time_stamp >= split]

            infatuations = filter_df(infatuations, self.config.infatuation_z_score)
            obsessions = filter_df(obsessions, self.config.obsession_z_score)

            # Infatuations have a max result limit
            infatuations = infatuations.iloc[
                : min(self.config.max_results_forgotten, len(infatuations))
            ]

            return infatuations, obsessions

        album_infatuations, album_obsessions = _get("album")
        artist_infatuations, artist_obsessions = _get("artist")
        infatuations = {"album": album_infatuations, "artist": artist_infatuations}
        obsessions = {"album": album_obsessions, "artist": artist_obsessions}
        return infatuations, obsessions

    def infatuations_data(self):
        if self.infatuations is None:
            self.infatuations, self.obsessions = self.infatuations_and_obsessions()
        return self.infatuations

    def obsessions_data(self):
        if self.obsessions is None:
            self.infatuations, self.obsessions = self.infatuations_and_obsessions()
        return self.obsessions

    def on_repeat_data(self):
        """Return the albums/artists that were listened to
        consecutively (irrespective of the time between each scrobble).
        """
        min_streak_length = self.config.min_streak_length

        def _get_streaks(attr):
            Obj = Artist if attr == "artist" else Album
            streaks = self.df.copy()
            streaks["start_of_streak"] = streaks[attr + "_name"].ne(
                streaks[attr + "_name"].shift()
            )
            streaks["streak_id"] = streaks["start_of_streak"].cumsum()
            streaks["streak_counter"] = streaks.groupby("streak_id").cumcount() + 1
            streaks["end_of_streak"] = streaks["start_of_streak"].shift(-1)
            streaks = streaks.iloc[1:-1]  # Remove NaNs
            streaks = streaks.rename(columns={"streak_counter": "streak_length"})
            streaks = streaks[streaks["end_of_streak"]]
            streaks = streaks[streaks["streak_length"] > min_streak_length]
            streaks = pd.DataFrame(
                [
                    {
                        **(
                            Obj._from_row(row)._to_row()
                        ),  # row is a Track, so convert to type Obj and serialise
                        "streak_length": row["streak_length"],
                    }
                    for _, row in streaks.iterrows()
                ]
            )
            df = streaks.sort_values(["streak_length"], ascending=False)
            df = df.drop_duplicates(subset=[attr + "_name"])
            return df.iloc[: min(self.config.max_results_repeat, len(df))]

        return {
            "album": _get_streaks("album"),
            "artist": _get_streaks("artist"),
        }

    def forgotten_data(self):
        """Return 'Forgotten' recommendations.
        These are albums/artists that were listened to
        once or twice a while ago, but not again since.
        Calculated using the 1st homology group of the
        album/artist timeseries, where forgotten
        albums/artists have their most persistent events
        at around 5-15 (averageish tracks per album),
        and only a few of them (listened to only a few
        times).
        """
        period = self.config.forget_period
        persistence_interval = self.config.persistence_interval
        n_session_thresh = self.config.n_session_thresh
        # To mask out events closer than `period` ago
        time_stamp_mask = np.asarray(self.bin_centres) < self._dt_to_int(TODAY - period)
        pi = persistence_interval
        isin = lambda pt: (pi[0] < pt[0] < pi[1]) and (pi[0] < pt[1] < pi[1])

        # Determine whether fast or accurate implementation should be used
        impl = "Fast"
        if FORCE_FORGOTTEN_FAST is False and len(self.df) < FORGOTTEN_FAST_SCROBBLE_THRESH:
            impl = 'Accurate'

        logging.warning(f"Using forgotten implementation={impl}.")

        def _get_forgotten(attr):
            forgotten = []
            for obj, timeseries in tqdm(self.timeseries[attr].items()):
                _ts = np.asarray(timeseries)[time_stamp_mask]
                if impl == "Accurate":
                    # This is much slow, but produces much great results
                    _data = np.asarray(list(enumerate(_ts))).reshape((-1, 2))
                    if _data.shape[0] == 0:
                        forgotten.append({**obj._to_row(), "value": 0})
                        continue
                    # Calling this for each obj is a bit slow. It's acceptable
                    # so long as the timeseries resoluthion isn't too high
                    hom = ripser(
                        _data, maxdim=1, n_perm=(len(_data) // RIPSER_N_PERM_FACTOR)
                    )["dgms"][1]
                    # hom = ripser(_data, maxdim=1)["dgms"][1]
                    if hom.shape[0] == 0:
                        forgotten.append({**obj._to_row(), "value": 0})
                        continue
                    # Filter sessions where only a few tracks were scrobbled
                    hom = hom[hom[:, 1] > pi[0]]
                    if hom.shape[0] == 0:
                        forgotten.append({**obj._to_row(), "value": 0})
                        continue
                    sessions = hom[np.apply_along_axis(isin, 1, hom)]
                    # If all points were within the persistence_interval, and there
                    # were only a few of them, then this album/artist is of interest
                    if (
                        hom.shape[0] == sessions.shape[0]
                        and sessions.shape[0] <= n_session_thresh
                    ):
                        forgotten.append({**obj._to_row(), "value": sessions.shape[0]})
                if impl == "Fast":
                    # This is very fast, and the results are okay
                    n_in = ((_ts <= pi[1]) & (_ts >= pi[0])).sum()
                    n_above = (_ts > pi[1]).sum()
                    if n_above > 0 or n_in == 0 or n_in + n_above > n_session_thresh:
                        forgotten.append({**obj._to_row(), "value": 0})
                        continue
                    forgotten.append({**obj._to_row(), "value": n_in})
            df = pd.DataFrame(forgotten).sort_values(["value"], ascending=True)
            df = df[df["value"] > 0]
            return df.iloc[: min(self.config.max_results_forgotten, len(df))]

        return {
            "album": _get_forgotten("album"),
            "artist": _get_forgotten("artist"),
        }


def run_engine(user: str):
    c = Cache(user)
    meta, data = c.read_scrobbles()
    engine = Engine(df=data, cache=c)
    return engine.run_engine()
