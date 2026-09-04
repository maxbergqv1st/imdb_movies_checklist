import os
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from data.loader import load_df
from pipeline import build_pipeline

load_dotenv()

WATCHED_FILE = Path("watched.json")
DEFAULT_CSV = Path(os.getenv("CSV_PATH", "movies.csv"))


def load_watched() -> set:
    if WATCHED_FILE.exists():
        return set(json.loads(WATCHED_FILE.read_text()))
    return set()


def save_watched(watched: set) -> None:
    WATCHED_FILE.write_text(json.dumps(list(watched)))


st.set_page_config(page_title="IMDB Checklist", layout="wide")
st.title("IMDB Movie Checklist")

# --- Data loading ---
if "df" not in st.session_state:
    if DEFAULT_CSV.exists():
        st.session_state.df = load_df(DEFAULT_CSV)
    else:
        uploaded = st.sidebar.file_uploader("Upload IMDB CSV", type="csv")
        if uploaded:
            st.session_state.df = load_df(uploaded)
        else:
            st.info("Drop `movies.csv` in this folder or upload via the sidebar.")
            st.stop()
else:
    st.sidebar.caption(f"{len(st.session_state.df)} titles loaded")

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
        view = view[view["genre"].apply(
            lambda x: any(g in str(x) for g in genre_filter) if __import__("pandas").notna(x) else False
        )]

    view = view.copy()
    view["watched"] = view["title"].isin(watched)

    edited = st.data_editor(
        view[["watched"] + display_cols],
        column_config={"watched": st.column_config.CheckboxColumn("✓")},
        use_container_width=True,
        hide_index=True,
    )

    for _, row in edited.iterrows():
        watched.add(row["title"]) if row["watched"] else watched.discard(row["title"])
    save_watched(watched)

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
