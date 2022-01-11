import dataclasses
import html
from typing import Dict
import os
import xml
from xml.etree import ElementTree


class Auth:
    @classmethod
    def get_public_key(cls):
        return os.getenv("LAST_FM_PUBLIC_KEY", "")

    @classmethod
    def get_secret_key(cls):
        return os.getenv("LAST_FM_PRIVATE_KEY", "")


def cleanup_nodes(doc):
    """
    Remove text nodes containing only whitespace
    """
    for node in doc.documentElement.childNodes:
        if node.nodeType == xml.dom.Node.TEXT_NODE and node.nodeValue.isspace():
            doc.documentElement.removeChild(node)
    return doc


def _number(string):
    """
    Extracts an int from a string.
    Returns a 0 if None or an empty string was passed.
    """

    if not string:
        return 0
    else:
        try:
            return int(string)
        except ValueError:
            return float(string)


def _unescape_htmlentity(string):
    mapping = html.entities.name2codepoint
    for key in mapping:
        string = string.replace("&%s;" % key, chr(mapping[key]))

    return string


def _extract(node, name=None, index=0):
    """Extracts a value from the xml string"""

    if name is not None:
        node = node.getElementsByTagName(name)
    else:
        node = [node]

    if len(node):
        if node[index].firstChild:
            return _unescape_htmlentity(node[index].firstChild.data.strip())
    else:
        return None


def _extract_list(node, name):
    """Extracts all the values from the xml string. returning a list."""
    return {
        idx: _extract(node, "image", idx)
        for idx, _ in enumerate(node.getElementsByTagName(name))
    }


def _get_cover(node, artist=False):
    if artist:
        # Artist images are children of the "artist" node
        all_covers = _extract_list(node.getElementsByTagName("artist")[0], "image")
        return all_covers[max(map(int, all_covers.keys()))]
    all_covers = node.getElementsByTagName("image")
    # Keep only images from this node (not children)
    all_covers = [n for n in all_covers if n.parentNode is node]
    all_covers = {idx: _extract(n) for idx, n in enumerate(all_covers)}
    return all_covers[max(map(int, all_covers.keys()))]


def _get_mbid(node, artist=False):
    if artist:
        return _extract(node.getElementsByTagName("artist")[0], "mbid")
    return node.getElementsByTagName("album")[0].getAttribute("mbid")


# The Track/Album/Artist objects should uniquely represent each of their
# associated real-world entities. This is difficult, since sometimes the
# same song may be included in multiple releases of the same album
# (e.g. Remastered releases), or inconsistent stylisation of an
# track/album/artist name (e.g. Artist "Spellling" is stylised as
# "SPELLLING"). Short of using MBIDs to determine matches, let's just use
# f"{track.lower()} - {album.lower()} - {artist.lower()}" as a unique key
# for each object. IMPORTANT: __eq__ and __hash__ need to implement the
# same key to ensure that, if two objects have a hash colission, then that
# means they are the same so long as they are __eq__.
# https://stackoverflow.com/a/9022664
@dataclasses.dataclass
class Artist:
    artist_name: str
    cover_uri: str
    mbid: str
    defined: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self):
        if self.artist_name is None:
            self.defined = False
            self.artist_name = ""

    def _unique_key(self):
        key = self.artist_name.lower()
        return key

    def __eq__(self, other):
        return self._unique_key() == other._unique_key()

    def __hash__(self):
        return hash(self._unique_key())

    def _to_row(self) -> Dict:
        return {
            "artist_name": self.artist_name,
            "cover_uri": self.cover_uri,
            "artist_mbid": self.mbid,
        }

    @classmethod
    def _from_row(cls, row):
        return Artist(
            artist_name=str(row["artist_name"]),
            cover_uri=row["cover_uri"],
            mbid=row["artist_mbid"],
        )


@dataclasses.dataclass
class Album:
    album_name: str
    artist: Artist
    cover_uri: str
    mbid: str
    defined: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self):
        if self.album_name is None:
            self.defined = False
            self.album_name = ""

    def _unique_key(self):
        key = " - ".join([self.album_name.lower(), self.artist.artist_name.lower()])
        return key

    def __eq__(self, other):
        return self._unique_key() == other._unique_key()

    def __hash__(self):
        return hash(self._unique_key())

    def _to_row(self) -> Dict:
        return {
            "album_name": self.album_name,
            "cover_uri": self.cover_uri,
            "album_mbid": self.mbid,
            **self.artist._to_row(),
        }

    @classmethod
    def _from_row(cls, row):
        return Album(
            album_name=str(row["album_name"]),
            artist=Artist._from_row(row),
            cover_uri=row["cover_uri"],
            mbid=row["album_mbid"],
        )


@dataclasses.dataclass
class Track:
    track_name: str
    time_stamp: int
    date_played: str
    loved: bool
    album: Album
    artist: Artist
    defined: bool = dataclasses.field(init=False, default=True)

    def __post_init__(self):
        if self.track_name is None:
            self.defined = False
            self.track_name = ""

    def _unique_key(self):
        key = " - ".join(
            [
                self.track_name.lower(),
                self.album.album_name.lower(),
                self.artist.artist_name.lower(),
            ]
        )
        return key

    def __eq__(self, other):
        return self._unique_key() == other._unique_key()

    def __hash__(self):
        return hash(self._unique_key())

    def _to_row(self) -> Dict:
        return {
            "track_name": self.track_name,
            "time_stamp": self.time_stamp,
            "date_played": self.date_played,
            "loved": self.loved,
            **self.album._to_row(),  # This includes artist information
        }

    @classmethod
    def _from_row(cls, row):
        album = Album._from_row(row)
        return Track(
            track_name=str(row["track_name"]),
            time_stamp=int(row["time_stamp"]),
            date_played=row["date_played"],
            loved=row["loved"],
            album=album,
            artist=album.artist,
        )


DUMMY_ARTIST = Artist(
    artist_name="",
    cover_uri="",
    mbid="",
)
DUMMY_ALBUM = Album(
    album_name="",
    artist=DUMMY_ARTIST,
    cover_uri="",
    mbid="",
)
DUMMY_TRACK = Track(
    track_name="",
    time_stamp="",
    date_played="",
    loved="",
    album=DUMMY_ALBUM,
    artist=DUMMY_ARTIST,
)
