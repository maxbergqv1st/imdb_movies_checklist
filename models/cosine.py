import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender, _build_text


class CosineRecommender(BaseRecommender):
    def fit(self, df: pd.DataFrame, watched: set) -> None:
        self._tfidf = TfidfVectorizer()
        matrix = self._tfidf.fit_transform(_build_text(df))
        watched_mask = df["title"].isin(watched).values
        self._profile = np.asarray(matrix[watched_mask].mean(axis=0))

    def score(self, df: pd.DataFrame) -> pd.Series:
        mat = self._tfidf.transform(_build_text(df))
        sims = cosine_similarity(self._profile, mat).flatten()
        return pd.Series(sims, index=df.index)
