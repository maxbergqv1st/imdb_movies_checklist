from abc import ABC, abstractmethod
import pandas as pd


def _build_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("genre",    pd.Series("", index=df.index)).fillna("") + " " +
        df.get("overview", pd.Series("", index=df.index)).fillna("")
    ).str.strip()


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, watched: set) -> None: ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return a score series aligned to df.index. Higher = more recommended."""
        ...
