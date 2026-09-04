import pandas as pd
from .base import BaseRecommender


class EnsembleRecommender(BaseRecommender):
    def __init__(self, models: list[tuple[BaseRecommender, float]]):
        self._models = models  # [(model, weight), ...]

    def fit(self, df: pd.DataFrame, watched: set) -> None:
        self._df = df
        self._watched = watched
        for model, _ in self._models:
            model.fit(df, watched)

    def score(self, df: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=df.index)
        for model, weight in self._models:
            s = model.score(df)
            rng = s.max() - s.min()
            total += weight * ((s - s.min()) / rng if rng > 0 else s)
        return total

    def recommend(self, n: int = 10) -> pd.DataFrame:
        unwatched = self._df[~self._df["title"].isin(self._watched)].copy()
        unwatched["score"] = self.score(self._df).loc[unwatched.index]
        return unwatched.nlargest(n, "score").drop(columns=["score"])
