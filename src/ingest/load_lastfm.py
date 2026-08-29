"""
Ingest Last.fm listening data into the unified cross-domain schema.

Downloads real artist play-count data (the Last.fm 360k dataset, ~184MB)
via implicit's dataset loader. Play counts are an implicit signal already
(nobody "rates" a song 1-5, they just play it or don't) -- unlike movies,
there's no thresholding step: any play count is a positive signal, weighted
by frequency.

Produces:
    data/processed/music_items.parquet
        item_id, domain, title (artist name)      -- ~5,000 rows, readable
    data/processed/music_interactions.parquet
        item_idx, user_idx, weight                -- millions of rows, ints only

Schema note: interactions store integer indices, not string IDs.
`implicit` needs an integer-indexed sparse matrix for training anyway, so
building "artist:12345" style string labels for millions of interaction
rows is wasted work in both directions -- it's slow, memory-heavy (numpy
stores unicode as fixed-width UCS4, so millions of short strings cost far
more than the integers they represent), and has to be un-done at training
time regardless. item_idx is a 0-based index directly aligned to row
position in the items table above, so no separate ID-to-index mapping
step is needed downstream.

This dataset does NOT include genre/mood tags -- needed for the
content-bridge step (src/models/content_embed.py) and must come from a
separate source (e.g. the Last.fm API), fetched locally since that API
isn't reachable from this sandbox. See scripts/fetch_music_tags.py.

Note on user_idx namespacing: these are raw Last.fm user indices, a
completely separate identity space from MovieLens user indices -- there
is no shared user overlap between domains (see load_movielens.py).
"""
import os
import numpy as np
import pandas as pd
from implicit.datasets.lastfm import get_lastfm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Cap to the top N most-played artists. Keeps the catalog fast to embed
# later, matches the scope the tag-fetching script needs to stay within
# Last.fm API rate limits, and keeps this pipeline runnable on a few GB of RAM.
TOP_N_ARTISTS = 5000


def load_lastfm():
    artists, users, plays = get_lastfm()  # csr_matrix: rows=artistId, cols=userId
    plays = plays.tocsr()

    # Rank artists by total play count, keep the top N, sorted so we can
    # slice the sparse matrix directly instead of scanning rows in Python.
    total_plays_per_artist = np.asarray(plays.sum(axis=1)).ravel()
    top_artist_idx = np.sort(np.argsort(total_plays_per_artist)[::-1][:TOP_N_ARTISTS])

    plays_top = plays[top_artist_idx, :].tocoo()
    # scipy's fancy-index slice re-indexes rows to a compact 0..N-1 space,
    # which conveniently already matches row position in `items` below --
    # no remapping needed, plays_top.row IS the item_idx.

    items = pd.DataFrame({
        "item_id": [f"artist:{i}" for i in top_artist_idx],  # only 5,000 rows, cheap
        "domain": "music",
        "title": artists[top_artist_idx],
    })

    interactions = pd.DataFrame({
        "item_idx": plays_top.row,
        "user_idx": plays_top.col,
        "weight": plays_top.data,
    })

    return items, interactions


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    items, interactions = load_lastfm()

    items.to_parquet(os.path.join(OUTPUT_DIR, "music_items.parquet"), index=False)
    interactions.to_parquet(os.path.join(OUTPUT_DIR, "music_interactions.parquet"), index=False)

    print(f"music: {len(items)} artists, {len(interactions)} interactions, "
          f"{interactions['user_idx'].nunique()} users")
