"""
Ingest MovieLens ratings into the unified cross-domain schema.

Uses the 20M dataset (ratings collected through ~2015), not the smaller
1M dataset (a fixed snapshot from 2000) -- the 1M version has literally
no movie made after 2000 in it, which made the demo's search feel
broken for anything modern. 20M still isn't fully current (no free, real
ratings dataset is), but it's a meaningfully better range.

The 20M pool has ~27,000 movies -- far more than needed, and slower at
every downstream step. So this caps to the TOP_N_MOVIES most-rated
titles, same pattern as TOP_N_ARTISTS in load_lastfm.py: keeps the
pipeline fast, and naturally favors well-known titles (including recent
ones -- popular modern movies accumulate ratings quickly) over obscure
long-tail ones nobody would search for in a demo anyway.

Ratings are binarized (>= 4 stars = positive implicit signal) so movies
share the same interaction format, CF training code, and evaluation
protocol as music.

Produces:
    data/processed/movies_items.parquet
        item_id, domain, title                    -- ~5,000 rows, readable
    data/processed/movies_interactions.parquet
        item_idx, user_idx, weight                 -- ints only, item_idx is a
                                                        0-based index aligned to
                                                        row position in items table

Schema matches load_lastfm.py -- see that file's docstring for why
interactions use integer indices rather than string IDs.

Note on user_idx namespacing: raw MovieLens user indices, a completely
separate identity space from Last.fm user indices -- there is no shared
cross-domain user overlap in the source data. Cross-domain bridging
happens later via content embeddings (src/models/content_embed.py).
"""
import os
import numpy as np
import pandas as pd
from implicit.datasets.movielens import get_movielens

POSITIVE_THRESHOLD = 4  # stars; ratings >= this count as a positive signal
TOP_N_MOVIES = 5000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def load_movielens(variant="20m"):
    titles, ratings = get_movielens(variant)  # csr_matrix: rows=movieId, cols=userId
    ratings = ratings.tocsr()

    # Some movie ids are unused padding rows (empty title) -- drop them,
    # remapping remaining ids to a compact 0..N-1 "valid" space.
    has_title = np.array([len(t) > 0 for t in titles])
    valid_orig_idx = np.nonzero(has_title)[0]

    orig_to_valid = np.full(len(titles), -1, dtype=np.int64)
    orig_to_valid[valid_orig_idx] = np.arange(len(valid_orig_idx))

    ratings = ratings.tocoo()
    valid_item = orig_to_valid[ratings.row]
    keep = valid_item >= 0
    valid_item, user_idx, rating_vals = valid_item[keep], ratings.col[keep], ratings.data[keep]

    is_positive = rating_vals >= POSITIVE_THRESHOLD
    valid_item, user_idx = valid_item[is_positive], user_idx[is_positive]

    # Cap to the top N most-rated movies (by positive interaction count)
    counts = np.bincount(valid_item, minlength=len(valid_orig_idx))
    top_valid_idx = np.sort(np.argsort(counts)[::-1][:TOP_N_MOVIES])

    valid_to_final = np.full(len(valid_orig_idx), -1, dtype=np.int64)
    valid_to_final[top_valid_idx] = np.arange(len(top_valid_idx))

    final_item = valid_to_final[valid_item]
    keep_final = final_item >= 0

    items = pd.DataFrame({
        "item_id": [f"movie:{valid_orig_idx[i]}" for i in top_valid_idx],
        "domain": "movie",
        "title": titles[valid_orig_idx[top_valid_idx]],
    })

    interactions = pd.DataFrame({
        "item_idx": final_item[keep_final],
        "user_idx": user_idx[keep_final],
        "weight": np.ones(int(keep_final.sum()), dtype=np.int64),  # already filtered to positive only
    })

    return items, interactions


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    items, interactions = load_movielens("20m")

    items.to_parquet(os.path.join(OUTPUT_DIR, "movies_items.parquet"), index=False)
    interactions.to_parquet(os.path.join(OUTPUT_DIR, "movies_interactions.parquet"), index=False)

    print(f"movies: {len(items)} items, {len(interactions)} positive interactions, "
          f"{interactions['user_idx'].nunique()} users")
