import json
from pathlib import Path

import pandas as pd

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

WATCHED_FILE = Path("watched.json")


def load_df(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})


def load_watched() -> set:
    if WATCHED_FILE.exists():
        return set(json.loads(WATCHED_FILE.read_text()))
    return set()


def save_watched(watched: set) -> None:
    WATCHED_FILE.write_text(json.dumps(list(watched)))
