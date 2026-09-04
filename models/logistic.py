import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .base import BaseRecommender


class LogisticRecommender(BaseRecommender):
    def _text(self, df: pd.DataFrame) -> pd.Series:
        return (
            df.get("genre", pd.Series("", index=df.index)).fillna("") + " " +
            df.get("overview", pd.Series("", index=df.index)).fillna("")
        ).str.strip()

    def _features(self, df: pd.DataFrame):
        text_f = self._tfidf.transform(self._text(df))
        if self._num_cols:
            num_f = self._scaler.transform(df[self._num_cols].fillna(0))
            return hstack([text_f, num_f])
        return text_f

    def fit(self, df: pd.DataFrame, watched: set) -> None:
        self._num_cols = [c for c in ["rating", "year", "votes"] if c in df.columns]

        self._tfidf = TfidfVectorizer(max_features=500)
        text_f = self._tfidf.fit_transform(self._text(df))

        if self._num_cols:
            self._scaler = StandardScaler(with_mean=False)
            num_f = self._scaler.fit_transform(df[self._num_cols].fillna(0))
            features = hstack([text_f, num_f])
        else:
            self._scaler = None
            features = text_f

        labels = df["title"].isin(watched).astype(int).values
        if labels.sum() == 0 or labels.sum() == len(labels):
            self._model = None
            return

        self._model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._model.fit(features, labels)

    def score(self, df: pd.DataFrame) -> pd.Series:
        if not getattr(self, "_model", None):
            return pd.Series(0.0, index=df.index)
        probs = self._model.predict_proba(self._features(df))[:, 1]
        return pd.Series(probs, index=df.index)
