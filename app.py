import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from data.loader import load_df, load_watched, save_watched
from pipeline import build_pipeline

load_dotenv()

DEFAULT_CSV = Path(os.getenv("CSV_PATH", "movies.csv"))

st.set_page_config(page_title="IMDB Checklist", layout="wide")
st.title("IMDB Movie Checklist")

# --- Data loading ---
if "df" not in st.session_state:
    if DEFAULT_CSV.exists():
        st.session_state.df = load_df(DEFAULT_CSV)
    else:
        st.info("Ingen data hittad. Ladda ner officiellt IMDB-dataset eller ladda upp en egen CSV.")
        col1, col2 = st.columns(2)

        if col1.button("Ladda ner IMDB-data (~15 000 filmer)", type="primary"):
            from data.download import fetch_movies
            with st.spinner("Hämtar från datasets.imdbws.com — tar ~30 sek..."):
                st.session_state.df = fetch_movies(str(DEFAULT_CSV))
            st.rerun()

        uploaded = col2.file_uploader("Eller ladda upp egen CSV", type="csv")
        if uploaded:
            st.session_state.df = load_df(uploaded)
            st.rerun()

        st.stop()
else:
    st.sidebar.caption(f"{len(st.session_state.df)} titlar inladdade")

df = st.session_state.df
watched = load_watched()
display_cols = [c for c in ["title", "year", "rating", "genre", "director", "runtime"] if c in df.columns]

tab1, tab2 = st.tabs(["All Movies", "Recommendations"])

# --- Tab 1: Checklist ---
with tab1:
    c1, c2, c3 = st.columns([3, 1, 1])
    search = c1.text_input("Search")
    show = c2.selectbox("Show", ["All", "Unwatched", "Watched"])

    genre_filter = []
    if "genre" in df.columns:
        genres = sorted({g.strip() for gs in df["genre"].dropna() for g in gs.split(",")})
        genre_filter = c3.multiselect("Genre", genres)

    view = df.copy()
    if search:
        view = view[view["title"].str.contains(search, case=False, na=False)]
    if show == "Watched":
        view = view[view["title"].isin(watched)]
    elif show == "Unwatched":
        view = view[~view["title"].isin(watched)]
    if genre_filter:
        pattern = "|".join(re.escape(g) for g in genre_filter)
        view = view[view["genre"].str.contains(pattern, na=False)]

    view["watched"] = view["title"].isin(watched)

    edited = st.data_editor(
        view[["watched"] + display_cols],
        column_config={"watched": st.column_config.CheckboxColumn("✓")},
        use_container_width=True,
        hide_index=True,
    )

    new_watched = set(edited.loc[edited["watched"], "title"])
    if new_watched != watched:
        save_watched(new_watched)
    watched = new_watched

    st.caption(f"{len(watched)} watched · {len(df) - len(watched)} remaining")

# --- Tab 2: Recommendations ---
with tab2:
    n = st.slider("How many?", 5, 50, 10)

    watched_hash = hash(frozenset(watched))
    if st.session_state.get("pipeline_hash") != watched_hash:
        with st.spinner("Fitting models..."):
            pipeline = build_pipeline()
            pipeline.fit(df, watched)
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_hash = watched_hash

    recs = st.session_state.pipeline.recommend(n)
    rec_cols = [c for c in display_cols if c in recs.columns]
    st.dataframe(recs[rec_cols] if rec_cols else recs, use_container_width=True, hide_index=True)
    st.caption("Cosine similarity (60%) + Logistic Regression (40%) · re-fits when watched list changes")
