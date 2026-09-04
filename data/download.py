"""
Laddar ner officiella IMDB-dataset från datasets.imdbws.com (ingen API-nyckel krävs).
Kombinerar title.basics + title.ratings och sparar som movies.csv.
"""
import gzip
import urllib.request
import pandas as pd

BASICS_URL  = "https://datasets.imdbws.com/title.basics.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

_BASICS_COLS = ["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres"]


def _fetch_basics() -> pd.DataFrame:
    with urllib.request.urlopen(BASICS_URL) as resp:
        with gzip.open(resp) as f:
            df = pd.read_csv(f, sep="\t", na_values="\\N", low_memory=False, usecols=_BASICS_COLS)
    return df[df["titleType"] == "movie"].drop(columns=["titleType"])


def _fetch_ratings() -> pd.DataFrame:
    with urllib.request.urlopen(RATINGS_URL) as resp:
        with gzip.open(resp) as f:
            return pd.read_csv(f, sep="\t", na_values="\\N")


def fetch_movies(out: str = "movies.csv", min_votes: int = 5_000) -> pd.DataFrame:
    print("Laddar basics...")
    movies = _fetch_basics()

    print("Laddar ratings...")
    ratings = _fetch_ratings()

    df = (
        movies.merge(ratings, on="tconst")
        .query("numVotes >= @min_votes")
        .rename(columns={
            "primaryTitle":   "title",
            "startYear":      "year",
            "runtimeMinutes": "runtime",
            "genres":         "genre",
            "averageRating":  "rating",
            "numVotes":       "votes",
        })
        .drop(columns=["tconst"])
        .reset_index(drop=True)
    )

    df.to_csv(out, index=False)
    print(f"Sparat {len(df)} filmer → {out}")
    return df


if __name__ == "__main__":
    fetch_movies()
