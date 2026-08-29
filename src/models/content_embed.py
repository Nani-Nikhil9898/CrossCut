"""
Embed movies and music into ONE shared semantic space using genre/mood
tags -- this is the actual cross-domain bridge, the hard part of the
original problem statement.

Why word vectors, not TF-IDF: movie genres ("Thriller", "Drama") and
music tags ("indie rock", "trip-hop") share almost no literal words, so
raw token overlap would barely connect the two domains at all. Word
vectors capture that "Thriller" is semantically close to "dark",
"suspenseful", "intense" even without exact word matches, which is what
actually lets a movie bridge to a thematically related artist.

Why spaCy's en_core_web_md instead of sentence-transformers: this
sandbox can reach GitHub (where spaCy's model wheels are hosted as
release assets) but not huggingface.co (where sentence-transformers
models live). Same idea -- both are pretrained embeddings averaged over
a short text description -- just a different, more reachable model.
This is a legitimate, documented tradeoff, not a hidden shortcut:
sentence-transformers would likely capture nuance a bit better, but
average-of-GloVe-vectors is a well-established baseline and is what's
actually running here.

Items with no matched tags (a small minority -- see VERIFY.md) fall back
to embedding their title instead of getting no signal at all.

Produces:
    data/processed/content_embeddings.npy    -- (N, 300) float32, L2-normalized
    data/processed/content_embed_items.parquet -- item_id, domain, title
                                                    row-aligned to the array above
"""
import os
import numpy as np
import pandas as pd
import spacy

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def build_text(row, tag_col):
    tags = row[tag_col]
    if isinstance(tags, str) and tags.strip():
        return tags
    return row["title"]  # fallback for the small minority with no matched tags


def embed_domain(nlp, items, tags, tag_col):
    merged = items.merge(tags, on="item_id", how="left")
    texts = merged.apply(lambda r: build_text(r, tag_col), axis=1)

    vectors = np.zeros((len(merged), nlp.vocab.vectors_length), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        vectors[i] = doc.vector

    return merged[["item_id", "domain", "title"]], vectors


def main():
    nlp = spacy.load("en_core_web_md")

    movies_items = pd.read_parquet(os.path.join(DATA_DIR, "movies_items.parquet"))
    movies_tags = pd.read_parquet(os.path.join(DATA_DIR, "movies_tags.parquet"))
    music_items = pd.read_parquet(os.path.join(DATA_DIR, "music_items.parquet"))
    music_tags = pd.read_parquet(os.path.join(DATA_DIR, "music_tags.parquet"))

    movie_meta, movie_vecs = embed_domain(nlp, movies_items, movies_tags, "genres")
    music_meta, music_vecs = embed_domain(nlp, music_items, music_tags, "tags")

    all_meta = pd.concat([movie_meta, music_meta], ignore_index=True)
    all_vecs = np.vstack([movie_vecs, music_vecs])

    # L2-normalize so a plain dot product IS cosine similarity downstream
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against empty/OOV text producing a zero vector
    all_vecs = all_vecs / norms

    np.save(os.path.join(DATA_DIR, "content_embeddings.npy"), all_vecs)
    all_meta.to_parquet(os.path.join(DATA_DIR, "content_embed_items.parquet"), index=False)

    print(f"embedded {len(movie_meta)} movies + {len(music_meta)} artists "
          f"= {len(all_meta)} items, {all_vecs.shape[1]}-dim shared space")

    return all_meta, all_vecs


if __name__ == "__main__":
    main()
