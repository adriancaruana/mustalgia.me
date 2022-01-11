import dataclasses
import logging
from pathlib import Path
import json
import os
from typing import Dict

import pandas as pd


CACHE_DIR = Path(os.getenv("LAST_FM_CACHE_DIR", ".")) / "cache"
DF = "df.csv.gz"
DF_PARQUET = "df_{part}.parquet"
META = "meta.json"


@dataclasses.dataclass
class Cache:
    user: str
    _cache_dir: Path = dataclasses.field(init=False, default=None)
    _meta: Dict = dataclasses.field(init=False, default=None)
    _df: pd.DataFrame = dataclasses.field(init=False, default=None)

    def __post_init__(self):

        self._cache_dir = CACHE_DIR / self.user
        self._cache_dir.mkdir(exist_ok=True, parents=True)

    @property
    def meta_path(self):
        return self._cache_dir / META

    @property
    def df_path(self):
        return self._cache_dir / DF

    @property
    def scrobbles_path(self):
        return self._cache_dir / "scrobbles"

    def key_to_path(self, key):
        return self._cache_dir / key

    def read_meta(self):
        return self.read_json(META)

    def write_scrobbles(self, meta: Dict, df: pd.DataFrame):
        logging.warning("Dumping to cache.")
        df.to_csv(self.df_path, index=False, compression="gzip")
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def write_scrobbles_part(self, meta: Dict, df: pd.DataFrame):
        logging.warning("Dumping to cache.")
        self.scrobbles_path.mkdir(exist_ok=True, parents=True)
        n = len(list(Path(self.scrobbles_path).glob("*")))
        path = self.scrobbles_path / DF_PARQUET.format(part=str(n).zfill(4))
        df.to_parquet(path, index=False)

        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def read_scrobbles(self):
        df = pd.read_csv(self.df_path)
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        return meta, df

    def read_scrobbles_parquet(self):
        df = pd.read_parquet(self.scrobbles_path)
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        return meta, df

    def write_df(self, key: str, df: pd.DataFrame):
        path = self.key_to_path(key) / DF
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False, compression="gzip")

    def read_df(self, key: str):
        return pd.read_csv((self.key_to_path(key) / DF))

    def write_json(self, key: str, data: Dict):
        with open(self.key_to_path(key), "w") as f:
            json.dump(data, f)

    def read_json(self, key: str):
        with open(self.key_to_path(key), "r") as f:
            return json.load(f)
