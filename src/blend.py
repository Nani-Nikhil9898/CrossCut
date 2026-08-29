"""
Blend CF-based and content-based signal into a cross-domain recommendation
engine -- this is where the two trained pieces (train_cf.py, content_embed.py)
actually come together.

Core design constraint: a real user of this system is cold-start BY
DEFINITION -- they don't exist in the historic MovieLens/Last.fm training
data, so there's no trained ALS *user* factor for them. Instead of
user-item CF prediction, this uses ITEM-item CF similarity (implicit's
built-in similar_items, comparing ALS item factors to each other) -- that
only needs the items a user says they like, not a trained user embedding,
so it works for a brand-new user with zero interaction history. This is
also, deliberately, the entire cold-start story: there's no separate
cold-start code path because nothing here ever required a trained user
in the first place.

Why CF similarity is same-domain only: movies' and music's ALS models
were trained completely independently (different data, different random
init), so their item-factor spaces have no reason to align with each
other -- a movie's CF vector and a song's CF vector are not comparable
numbers. Only the content embedding space (content_embed.py) is shared
across domains, so cross-domain scoring ALWAYS goes through content
similarity, never CF.

Why recommendations are returned as two separate per-domain lists
instead of one merged ranked list: same-domain candidates get a CF +
content blend, cross-domain candidates get content only -- those scores
aren't necessarily on a directly comparable footing for ranking purposes,
even though both are bounded cosine similarities. Keeping movies and
music as separate ranked lists sidesteps that comparability question
entirely, and matches how real systems present this anyway (Spotify's
"more of this" vs. "you might also like" are separate rails, not one
interleaved list).
"""
import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")

CF_WEIGHT = 0.7
CONTENT_WEIGHT = 0.3

DOMAIN_TAG = {"movies": "movie", "music": "music"}  # domain-table name -> content `domain` value


class TasteEngine:
    def __init__(self):
        self.cf = {}
        for domain in ("movies", "music"):
            with open(os.path.join(MODEL_DIR, f"{domain}_cf.pkl"), "rb") as f:
                self.cf[domain] = pickle.load(f)["model"]

        self.domain_items = {
            d: pd.read_parquet(os.path.join(DATA_DIR, f"{d}_items.parquet"))
            for d in ("movies", "music")
        }
        # item_id -> local row index, built explicitly per domain rather than
        # assumed positional, so this stays correct even if an upstream merge
        # ever reorders rows.
        self.local_idx = {
            d: dict(zip(df["item_id"], df.index)) for d, df in self.domain_items.items()
        }

        self.content_items = pd.read_parquet(os.path.join(DATA_DIR, "content_embed_items.parquet"))
        self.content_vecs = np.load(os.path.join(DATA_DIR, "content_embeddings.npy"))
        self.content_idx = dict(zip(self.content_items["item_id"], self.content_items.index))

    def _content_vec(self, item_id):
        return self.content_vecs[self.content_idx[item_id]]

    def _content_sim_for_domain(self, item_id, target_domain):
        """Content similarity of one seed to every item in target_domain,
        returned as an array aligned to self.domain_items[target_domain] rows."""
        target_tag = DOMAIN_TAG[target_domain]
        target_rows = self.content_items[self.content_items["domain"] == target_tag]
        sims = self.content_vecs[target_rows.index] @ self._content_vec(item_id)
        sim_by_id = dict(zip(target_rows["item_id"], sims))
        items = self.domain_items[target_domain]
        return items["item_id"].map(sim_by_id).fillna(0.0).to_numpy()

    def _cf_sim_for_domain(self, item_id, domain):
        """Item-item CF similarity of one seed to every item in ITS OWN
        domain (never cross-domain -- see module docstring), aligned to
        self.domain_items[domain] rows."""
        local_i = self.local_idx[domain][item_id]
        n_items = len(self.domain_items[domain])
        ids, scores = self.cf[domain].similar_items(local_i, N=n_items)
        sim_by_idx = dict(zip(ids.tolist(), scores.tolist()))
        return np.array([sim_by_idx.get(i, 0.0) for i in range(n_items)])

    def recommend(self, seed_item_ids, top_k=10):
        """
        seed_item_ids: item_ids the user says they like, e.g.
        ["movie:1", "artist:21"] -- mixing domains freely is the whole point.

        Returns {"movies": DataFrame, "music": DataFrame}, each the top_k
        items in that domain ranked by blended score, seeds excluded.
        """
        results = {}
        for target_domain in ("movies", "music"):
            target_tag = DOMAIN_TAG[target_domain]
            items = self.domain_items[target_domain]
            scores = np.zeros(len(items))

            for seed_id in seed_item_ids:
                seed_tag = self.content_items.loc[self.content_idx[seed_id], "domain"]
                content_sim = self._content_sim_for_domain(seed_id, target_domain)

                if seed_tag == target_tag:
                    cf_sim = self._cf_sim_for_domain(seed_id, target_domain)
                    scores += CF_WEIGHT * cf_sim + CONTENT_WEIGHT * content_sim
                else:
                    scores += content_sim  # cross-domain: content is the only bridge there is

            scores /= len(seed_item_ids)

            ranked = items.copy()
            ranked["score"] = scores
            ranked = ranked[~ranked["item_id"].isin(seed_item_ids)]
            results[target_domain] = ranked.sort_values("score", ascending=False).head(top_k)

        return results


if __name__ == "__main__":
    engine = TasteEngine()
    recs = engine.recommend(["movie:1"])  # Toy Story, no music seed at all
    print("Seed: Toy Story only\n")
    print("Movies:\n", recs["movies"][["title", "score"]].to_string(index=False))
    print("\nMusic:\n", recs["music"][["title", "score"]].to_string(index=False))
