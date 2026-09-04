import os
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

WATCHED_FILE = Path("watched.json")
DEFAULT_CSV = Path(os.getenv("CSV_PATH", "movies.csv"))

COLUMN_ALIASES = {
    "Series_Title": "title", "Title": "title", "name": "title",
    "Released_Year": "year", "Year": "year",
    "IMDB_Rating": "rating", "Rating": "rating", "imdb_rating": "rating",
    "Genre": "genre",
    "Overview": "overview", "Description": "overview", "description": "overview",
    "Director": "director",
    "Runtime": "runtime",
    "No_of_Votes": "votes", "Votes": "votes",
}


def load_df(source):
    df = pd.read_csv(source)
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})


def load_watched():
    if WATCHED_FILE.exists():
        return set(json.loads(WATCHED_FILE.read_text()))
    return set()


def save_watched(watched):
    WATCHED_FILE.write_text(json.dumps(list(watched)))


def recommend(df, watched, n):
    unwatched = df[~df["title"].isin(watched)].copy()
    if not watched or "genre" not in df.columns:
        return unwatched.nlargest(n, "rating") if "rating" in unwatched.columns else unwatched.head(n)

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(df["genre"].fillna(""))

    watched_mask = df["title"].isin(watched)
    profile = matrix[watched_mask.values].mean(axis=0)

    unwatched_idx = (~watched_mask).values
    sims = cosine_similarity(profile, matrix[unwatched_idx]).flatten()

    unwatched = unwatched.copy()
    unwatched["_sim"] = sims

    if "rating" in unwatched.columns:
        r = unwatched["rating"]
        rng = r.max() - r.min()
        r_norm = (r - r.min()) / rng if rng else 0
        unwatched["_score"] = 0.6 * unwatched["_sim"] + 0.4 * r_norm
    else:
        unwatched["_score"] = unwatched["_sim"]

    return unwatched.nlargest(n, "_score").drop(columns=["_sim", "_score"])


st.set_page_config(page_title="IMDB Checklist", layout="wide")
st.title("IMDB Movie Checklist")

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
            lambda x: any(g in str(x) for g in genre_filter) if pd.notna(x) else False
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

with tab2:
    n = st.slider("How many?", 5, 50, 10)
    recs = recommend(df, watched, n)
    rec_cols = [c for c in display_cols if c in recs.columns]
    st.dataframe(recs[rec_cols] if rec_cols else recs, use_container_width=True, hide_index=True)
    st.caption("Ranked by genre overlap with your watched history, then by IMDB rating")
