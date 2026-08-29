"""
Train a collaborative-filtering model for one domain using implicit ALS.

Deliberately domain-agnostic: this function doesn't know or care whether
it's training on movies or music, it just needs a (user_idx, item_idx,
weight) interaction table. That's the whole point of standardizing both
domains onto the same schema in the ingest layer -- one training path
instead of duplicated per-domain logic.

Usage:
    python3 src/models/train_cf.py movies
    python3 src/models/train_cf.py music
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")

FACTORS = 64
REGULARIZATION = 0.05
ITERATIONS = 20
RANDOM_STATE = 42


def build_matrix(interactions, n_items, n_users):
    """user-item sparse matrix, shape (n_users, n_items) -- what implicit expects."""
    return sp.csr_matrix(
        (interactions["weight"].astype(np.float32),
         (interactions["user_idx"], interactions["item_idx"])),
        shape=(n_users, n_items),
    )


def leave_last_out_split(interactions):
    """
    Standard implicit-feedback eval protocol: for each user, hold out ONE
    interaction as the test item, train on the rest. A random row-level
    split would leak information (a user's test item could trivially
    resemble their train items) and wouldn't actually test ranking quality.

    For music, "last" isn't meaningful (no timestamps in this dataset,
    just play counts), so we hold out each user's single highest-weight
    interaction instead -- their clearest positive signal, which is the
    right analogue for movies' latest 5-star-equivalent rating.

    Users with only 1 interaction are dropped entirely: they can't supply
    both a train example and a held-out test example. In production
    they'd get the content-based cold-start path instead of CF.
    """
    counts = interactions.groupby("user_idx").size()
    eligible = counts[counts >= 2].index
    interactions = interactions[interactions["user_idx"].isin(eligible)]

    interactions = interactions.sample(frac=1, random_state=RANDOM_STATE)  # shuffle ties
    test_idx = (
        interactions.sort_values("weight", ascending=False)
        .groupby("user_idx", sort=False)
        .head(1)
        .index
    )
    test = interactions.loc[test_idx]
    train = interactions.drop(test_idx)
    return train, test


def train_domain(domain, factors=FACTORS, iterations=ITERATIONS):
    items = pd.read_parquet(os.path.join(DATA_DIR, f"{domain}_items.parquet"))
    interactions = pd.read_parquet(os.path.join(DATA_DIR, f"{domain}_interactions.parquet"))

    n_items = len(items)
    n_users = int(interactions["user_idx"].max()) + 1

    train, test = leave_last_out_split(interactions)

    train_matrix = build_matrix(train, n_items, n_users)

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=REGULARIZATION,
        iterations=iterations,
        random_state=RANDOM_STATE,
    )
    model.fit(train_matrix)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{domain}_cf.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "train_matrix": train_matrix,  # needed at inference to exclude seen items
            "test": test,
            "n_items": n_items,
            "n_users": n_users,
        }, f)

    print(f"[{domain}] trained: {n_users} users, {n_items} items, "
          f"{train_matrix.nnz} train interactions, {len(test)} held out for eval")
    return model, train_matrix, test


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "movies"
    train_domain(domain)
