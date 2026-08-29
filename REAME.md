# Crosscut

A recommendation system that works across movies and music at the same time, instead of treating them as two separate apps. Pick a few movies or artists you like, and it recommends more of both - including things in the *other* domain, based on a shared idea of what your taste actually is.

Scoped down from a broader hackathon prompt (movies, music, podcasts, video, news) to just movies and music - those are the two domains with solid public interaction datasets to train real collaborative filtering on. News was considered and specifically ruled out; see Known limitations below for why.

## What it actually does

- Search for a movie or artist, add a few you like
- Get recommendations back in **both** domains:
  - same-domain picks come from real collaborative filtering (people who liked what you liked also liked...)
  - cross-domain picks come from a shared embedding space built out of genre/mood tags, since there's no dataset anywhere with the same person's movie ratings and listening history side by side

As a sanity check while building it: seeding with *The Dark Knight* surfaces *Batman Begins* on the movie side (its own prequel) and dark/aggressive bands like Sunn O and Charles Bronson on the music side, purely from shared "intense, dark" tags with zero literal word overlap between "Thriller" and any band name.

## How it's built

- **Movies**: MovieLens 20M ratings, capped to the 5,000 most-rated titles (tops out around 2014 - there's no free ratings dataset that goes more current than that)
- **Music**: Last.fm's 360K-user dataset, capped to the 5,000 most-played artists (collected in 2008, so nothing after that exists in the catalog)
- **Collaborative filtering**: `implicit`'s ALS, trained separately per domain (9.6M real positive movie interactions, 12.6M real music plays)
- **Content bridge**: genre/mood tags pulled from TMDB and the Last.fm API, embedded into a shared 300-dim word-vector space so a movie and a song can actually be compared
- **Blending**: same-domain scores mix CF + content similarity; cross-domain scores are content-only, since the two ALS models were trained completely independently and their vector spaces aren't comparable to each other
- **Serving**: FastAPI backend, Streamlit demo UI, both load a precomputed model rather than retraining on request

## Stack

Python, `implicit`, spaCy word vectors (only used to *generate* the embeddings - the runtime app never touches spaCy, it just loads the resulting numpy file), FastAPI, Streamlit. Dependency management via `uv`.

## Running it

```bash
uv sync

uv run src/ingest/load_movielens.py
uv run src/ingest/load_lastfm.py

uv run src/models/train_cf.py movies
uv run src/models/train_cf.py music

# needs movies_tags.parquet / music_tags.parquet -- see scripts/fetch_movie_tags.py
# and scripts/fetch_music_tags.py, both need a free API key (instructions in
# the docstring at the top of each)
uv run src/models/content_embed.py

uv run src/blend.py          # quick sanity check in the terminal
uv run uvicorn api.main:app --reload   # REST API, docs at /docs
uv run streamlit run app/demo.py       # the actual demo UI
```

<!-- Every step above has its expected output written down in `VERIFY.md`, so you can check your own run against what it's supposed to produce instead of just hoping it worked. -->

## Does it actually work?

Evaluated with leave-one-out hit rate and NDCG on real held-out data, not just eyeballed output:

| Domain | HR@5  | HR@10 | HR@20 | NDCG@10 |
| ------ | ----- | ----- | ----- | ------- |
| Movies | 0.145 | 0.212 | 0.299 | 0.121   |
| Music  | 0.246 | 0.353 | 0.480 | 0.204   |

Random guessing against a 5,000-item catalog would land around HR@10 ≈ 0.002.

## Known limitations

Listing these on purpose instead of hoping nobody notices:

- **Neither dataset is current.** Movies stop around 2014, music stops in 2008.
- **The cross-domain bridge only has tags to work with, not real shared behavior.** No platform publishes a dataset where the same person's movie ratings and music history are linked - not even Netflix and Spotify share that with each other. Content similarity is the honest answer to that gap, not a shortcut around it.
- **Sparse tags produce weaker matches.** A movie tagged with just "Crime, Drama" (two generic words) sometimes drifts toward comedians in the cross-domain recommendations - happened with both *Casino* and *American Gangster* independently, so it's a real, repeatable pattern, not a fluke.
- **News was considered and dropped.** News articles go stale in hours or days, so a model trained on old click logs would recommend obviously outdated articles in a live demo, and new articles have severe cold-start the moment they're published - a real, known limitation in production news recommenders, not something specific to this project.

## What I'd add next

- A proper quantitative check on cross-domain match quality, not just spot-checking examples by hand
- A lightweight feedback signal so a user's clicks nudge their own future recommendations
- Deploying the Streamlit app somewhere public instead of only running it locally
