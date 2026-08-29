"""
Streamlit demo UI for the cross-domain taste engine.

Imports TasteEngine directly (not via the FastAPI layer) so this runs as
one self-contained app -- simplest path to a deployable demo link (e.g.
Streamlit Community Cloud), no separate API server to manage. api/main.py
exists independently as the "this is also a real API" piece.

Picks are kept in st.session_state rather than a plain multiselect, since
a multiselect's option list would change every time the search box
changes -- Streamlit silently drops selections that fall outside the
current options, which would make earlier picks vanish as soon as you
searched for something else. session_state avoids that entirely.

Run it:
    uv run streamlit run app/demo.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from blend import TasteEngine  # noqa: E402

st.set_page_config(page_title="Taste Engine", layout="wide")


@st.cache_resource
def load_engine():
    return TasteEngine()


engine = load_engine()

if "seeds" not in st.session_state:
    st.session_state.seeds = {}  # item_id -> (domain, title)

st.title("Cross-domain taste engine")
st.caption(
    "Real collaborative filtering trained on MovieLens and Last.fm data, "
    "blended with a shared content-embedding bridge -- so a movie you "
    "like can genuinely surface a matching artist, and vice versa."
)

st.subheader("1. Pick a few things you like")
search_col, domain_col = st.columns([3, 1])
query = search_col.text_input(
    "Search movies or artists", label_visibility="collapsed",
    placeholder="Search movies or artists...",
)
domain_choice = domain_col.selectbox(
    "Domain", ["Both", "Movies", "Music"], label_visibility="collapsed",
)

if query:
    domains = (
        ["movies", "music"] if domain_choice == "Both"
        else ["movies"] if domain_choice == "Movies"
        else ["music"]
    )
    for d in domains:
        items = engine.domain_items[d]
        matches = items[items["title"].str.contains(query, case=False, na=False, regex=False)].head(8)
        for _, row in matches.iterrows():
            item_id, title = row["item_id"], row["title"]
            c1, c2 = st.columns([5, 1])
            c1.write(f"{title}  \u00b7  _{d}_")
            if item_id in st.session_state.seeds:
                c2.button("Added", key=f"add_{item_id}", disabled=True)
            elif c2.button("Add", key=f"add_{item_id}"):
                st.session_state.seeds[item_id] = (d, title)
                st.rerun()

if st.session_state.seeds:
    st.subheader("Your picks")
    for item_id, (d, title) in list(st.session_state.seeds.items()):
        c1, c2 = st.columns([5, 1])
        c1.write(f"{title}  \u00b7  _{d}_")
        if c2.button("Remove", key=f"rm_{item_id}"):
            del st.session_state.seeds[item_id]
            st.rerun()

st.subheader("2. Get recommendations")
if st.button("Recommend", disabled=len(st.session_state.seeds) == 0, type="primary"):
    with st.spinner("Blending collaborative filtering and content signal..."):
        recs = engine.recommend(list(st.session_state.seeds.keys()), top_k=10)

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("### Movies for you")
        for row in recs["movies"].itertuples():
            st.write(f"**{row.title}**")
            st.progress(min(max(row.score, 0.0), 1.0))
    with rcol2:
        st.markdown("### Music for you")
        for row in recs["music"].itertuples():
            st.write(f"**{row.title}**")
            st.progress(min(max(row.score, 0.0), 1.0))
elif len(st.session_state.seeds) == 0:
    st.info("Add at least one movie or artist above to get recommendations.")
