"""
Fetch top tags (genre/mood folksonomy) for artists from the Last.fm API.

Run this on YOUR machine, not in my sandbox -- ws.audioscrobbler.com isn't
reachable from there. Needs a free Last.fm API key:
    1. Create an account: https://www.last.fm/join
    2. Get a key: https://www.last.fm/api/account/create (instant, free)

Setup:
    pip install requests pandas pyarrow
    export LASTFM_API_KEY=your_key_here
    python3 scripts/fetch_music_tags.py

Reads:  data/processed/music_items.parquet   (item_id, domain, title = artist name)
Writes: data/processed/music_tags.parquet    (item_id, tags)

Only 5,000 artists (see TOP_N_ARTISTS in src/ingest/load_lastfm.py), so
even at a polite delay this comfortably finishes in under 30 minutes and
stays well inside Last.fm's rate limit. Prints progress every 200 artists.
"""
import truststore
truststore.inject_into_ssl()  # see fetch_movie_tags.py -- same fix, same reason

import os
import sys
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

API_KEY = os.environ.get("LASTFM_API_KEY")
API_URL = "http://ws.audioscrobbler.com/2.0/"
REQUEST_DELAY = 0.25  # Last.fm asks for <=5 req/sec; this stays well under that
TOP_TAGS_PER_ARTIST = 5


def make_session():
    """See fetch_movie_tags.py's make_session() for why this exists --
    same fix for the same class of connection-reset issue."""
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = make_session()


def fetch_tags(artist_name):
    params = {
        "method": "artist.gettoptags",
        "artist": artist_name,
        "api_key": API_KEY,
        "format": "json",
        "autocorrect": 1,
    }
    resp = SESSION.get(API_URL, params=params, timeout=10)
    resp.raise_for_status()
    tags = resp.json().get("toptags", {}).get("tag", [])
    return [t["name"] for t in tags[:TOP_TAGS_PER_ARTIST]]


def main():
    if not API_KEY:
        sys.exit("Set LASTFM_API_KEY first -- see the docstring at the top of this file.")

    items = pd.read_parquet("data/processed/music_items.parquet")

    rows = []
    for i, row in items.iterrows():
        try:
            tags = fetch_tags(row["title"])
        except Exception as e:
            print(f"  [warn] {row['title']}: {e}")
            tags = []
        rows.append({"item_id": row["item_id"], "tags": ", ".join(tags)})
        if i % 200 == 0:
            print(f"{i}/{len(items)}  {row['title']} -> {tags}")
        time.sleep(REQUEST_DELAY)

    out = pd.DataFrame(rows)
    out.to_parquet("data/processed/music_tags.parquet", index=False)
    matched = (out["tags"] != "").sum()
    print(f"\nDone: {matched}/{len(out)} artists matched with tags.")
    print("Send back: data/processed/music_tags.parquet")


if __name__ == "__main__":
    main()
