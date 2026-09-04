from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, watched: set) -> None: ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return a score series aligned to df.index. Higher = more recommended."""
        ...
