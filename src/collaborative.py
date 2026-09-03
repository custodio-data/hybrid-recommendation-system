"""
collaborative.py
-----------------
Item-based Collaborative Filtering: recommends movies based on rating
patterns across users (as opposed to the movies' own content).
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_ratings_matrix(ratings_train: pd.DataFrame) -> pd.DataFrame:
    """Pivot the training ratings into a userId x movieId matrix."""
    return ratings_train.pivot_table(index="userId", columns="movieId", values="rating")


def normalize_ratings(ratings_matrix: pd.DataFrame) -> pd.DataFrame:
    """Mean-center each user's ratings (removes each user's rating bias,
    e.g. someone who always rates 5 vs. someone who rates around 3)."""
    return ratings_matrix.sub(ratings_matrix.mean(axis=1), axis=0)


def build_item_similarity(ratings_norm: pd.DataFrame) -> pd.DataFrame:
    """Item-item cosine similarity from normalized ratings.

    NaNs (unrated movies) are filled with 0 only for this similarity
    calculation — the mean-centered ratings themselves stay untouched.
    """
    similarity = cosine_similarity(ratings_norm.fillna(0).T)
    return pd.DataFrame(similarity, index=ratings_norm.columns, columns=ratings_norm.columns)


def build_collaborative_model(ratings_train: pd.DataFrame) -> pd.DataFrame:
    """Run the full CF pipeline and return the item_similarity_cf matrix
    (index/columns = movieId), ready to be combined with content-based
    scores in hybrid.py.
    """
    ratings_matrix = build_ratings_matrix(ratings_train)
    ratings_norm = normalize_ratings(ratings_matrix)
    return build_item_similarity(ratings_norm)
