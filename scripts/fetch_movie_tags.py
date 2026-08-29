"""
Fetch genre tags for movies from TMDB (The Movie Database).

Run this on YOUR machine, not in my sandbox -- api.themoviedb.org isn't
reachable from there (only pypi/npm/github-style hosts are). Needs a free
TMDB API key:
    1. Create an account: https://www.themoviedb.org/signup
    2. Settings -> API -> request a key (instant, free, no approval wait)

Setup:
    pip install requests pandas pyarrow
    export TMDB_API_KEY=your_key_here
    python3 scripts/fetch_movie_tags.py

Reads:  data/processed/movies_items.parquet   (item_id, domain, title)
Writes: data/processed/movies_tags.parquet    (item_id, genres)

MovieLens titles look like "Toy Story (1995)" -- the year is parsed out
and passed to TMDB's search to disambiguate remakes/re-releases.

~3,900 movies at a polite request rate takes roughly 10-15 minutes,
mostly network latency, not the delay itself. Safe to run in the
background; it prints progress every 200 movies so you can see it's alive.
"""
import truststore
truststore.inject_into_ssl()  # use Windows' native trust store instead of Python's own --
                               # fixes TLS handshake resets on networks/AV that only
                               # recognize native Windows TLS clients (see curl.exe test)

import os
import re
import sys
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

API_KEY = os.environ.get("TMDB_API_KEY")
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
GENRE_LIST_URL = "https://api.themoviedb.org/3/genre/movie/list"
REQUEST_DELAY = 0.05  # TMDB's free tier allows far more than this; just being polite

TITLE_YEAR_RE = re.compile(r"^(.*)\s\((\d{4})\)\s*$")


def make_session():
    """
    A plain requests.get() reuses a global connection pool that can hand
    back a connection the server (or something in between -- antivirus
    HTTPS inspection, a VPN, a corporate proxy) has already closed,
    surfacing as WinError 10054 / ConnectionResetError. Retrying with
    backoff on a session recovers from that automatically instead of
    giving up on the first hiccup.
    """
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.5,  # waits ~1.5s, 3s, 6s, 12s, 24s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = make_session()


def parse_title(raw_title):
    m = TITLE_YEAR_RE.match(raw_title)
    if m:
        return m.group(1).strip(), m.group(2)
    return raw_title.strip(), None


def fetch_genre_ids(title, year):
    params = {"api_key": API_KEY, "query": title}
    if year:
        params["year"] = year
    resp = SESSION.get(SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0].get("genre_ids", []) if results else []


def main():
    if not API_KEY:
        sys.exit("Set TMDB_API_KEY first -- see the docstring at the top of this file.")

    items = pd.read_parquet("data/processed/movies_items.parquet")

    genre_map = {
        g["id"]: g["name"]
        for g in SESSION.get(GENRE_LIST_URL, params={"api_key": API_KEY}, timeout=10)
        .json()["genres"]
    }

    rows = []
    for i, row in items.iterrows():
        title, year = parse_title(row["title"])
        try:
            genre_ids = fetch_genre_ids(title, year)
        except Exception as e:
            print(f"  [warn] {title}: {e}")
            genre_ids = []
        genres = [genre_map[g] for g in genre_ids if g in genre_map]
        rows.append({"item_id": row["item_id"], "genres": ", ".join(genres)})
        if i % 200 == 0:
            print(f"{i}/{len(items)}  {title} -> {genres}")
        time.sleep(REQUEST_DELAY)

    out = pd.DataFrame(rows)
    out.to_parquet("data/processed/movies_tags.parquet", index=False)
    matched = (out["genres"] != "").sum()
    print(f"\nDone: {matched}/{len(out)} movies matched with genres.")
    print("Send back: data/processed/movies_tags.parquet")


if __name__ == "__main__":
    main()
