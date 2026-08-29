"""
Offline evaluation via the leave-last-out protocol: Hit Rate@K and NDCG@K.

For each user, one interaction was held out at training time (see
train_cf.py's leave_last_out_split). This checks whether that held-out
item shows up in the model's top-K recommendations for that user.

  HR@K:   fraction of users whose held-out item appears in their top-K.
  NDCG@K: same idea, but rewards ranking it higher within the top-K
          (1/log2(rank+2), the standard discounted-gain formula).

This is the standard "leave-one-out" evaluation protocol used in
implicit-feedback recsys literature (e.g. He et al., Neural Collaborative
Filtering, 2017) -- not an ad hoc metric invented for this project.

Usage:
    python3 src/models/evaluate.py movies
    python3 src/models/evaluate.py music
"""
import os
import sys
import pickle
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
K_VALUES = [5, 10, 20]


def evaluate(domain):
    with open(os.path.join(MODEL_DIR, f"{domain}_cf.pkl"), "rb") as f:
        d = pickle.load(f)
    model, train_matrix, test = d["model"], d["train_matrix"], d["test"]

    test_users = test["user_idx"].to_numpy()
    target_by_user = dict(zip(test["user_idx"], test["item_idx"]))

    max_k = max(K_VALUES)
    rec_ids, _ = model.recommend(
        test_users, train_matrix[test_users], N=max_k, filter_already_liked_items=True
    )

    hits = {k: 0 for k in K_VALUES}
    ndcgs = {k: 0.0 for k in K_VALUES}
    n = len(test_users)

    for row, user in enumerate(test_users):
        target = target_by_user[user]
        pos = np.where(rec_ids[row] == target)[0]
        rank = int(pos[0]) if len(pos) else None
        for k in K_VALUES:
            if rank is not None and rank < k:
                hits[k] += 1
                ndcgs[k] += 1.0 / np.log2(rank + 2)

    print(f"\n[{domain}] leave-last-out evaluation, {n} held-out users:")
    for k in K_VALUES:
        print(f"  HR@{k:<3d} = {hits[k] / n:.4f}    NDCG@{k:<3d} = {ndcgs[k] / n:.4f}")

    return {f"HR@{k}": hits[k] / n for k in K_VALUES} | {f"NDCG@{k}": ndcgs[k] / n for k in K_VALUES}


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "movies"
    evaluate(domain)
