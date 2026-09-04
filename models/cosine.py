import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender


class CosineRecommender(BaseRecommender):
    def fit(self, df: pd.DataFrame, watched: set) -> None:
        text = (
            df.get("genre", pd.Series("", index=df.index)).fillna("") + " " +
            df.get("overview", pd.Series("", index=df.index)).fillna("")
        ).str.strip()
        self._tfidf = TfidfVectorizer()
        self._matrix = self._tfidf.fit_transform(text)
        watched_mask = df["title"].isin(watched).values
        self._profile = self._matrix[watched_mask].mean(axis=0)

    def score(self, df: pd.DataFrame) -> pd.Series:
        sims = cosine_similarity(self._profile, self._matrix).flatten()
        return pd.Series(sims, index=df.index)
