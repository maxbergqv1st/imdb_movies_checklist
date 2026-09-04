from models.cosine import CosineRecommender
from models.logistic import LogisticRecommender
from models.ensemble import EnsembleRecommender


def build_pipeline() -> EnsembleRecommender:
    return EnsembleRecommender([
        (CosineRecommender(), 0.6),
        (LogisticRecommender(), 0.4),
    ])
