"""Shared model definition so training, serving and analysis use one pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config

FEATURES = config.NUMERIC + config.CATEGORICAL + [config.TEXT]


def build_pipeline(n_estimators=300, max_depth=None, seed=42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
        ("txt", TfidfVectorizer(max_features=150, ngram_range=(1, 2), min_df=3), config.TEXT),
    ])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                class_weight="balanced", n_jobs=-1, random_state=seed)
    return Pipeline([("prep", pre), ("rf", rf)])
