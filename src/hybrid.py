"""
hybrid.py
---------
Combines Content-Based (CB) and Collaborative Filtering (CF) scores into a
single hybrid score:

    hybrid_score = alpha * CB_score + (1 - alpha) * CF_score

alpha=1.0 -> pure content-based · alpha=0.0 -> pure collaborative.

Also includes the Precision@k / Recall@k evaluation used to pick alpha.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Index helper
# --------------------------------------------------------------------------- #

def build_id_index(movies_df: pd.DataFrame) -> Dict[int, int]:
    """Map movieId -> row position in movies_df / cb_similarity.

    Needed because CB similarity is indexed by DataFrame row position,
    while CF similarity (from collaborative.py) is indexed by movieId —
    this dict lets hybrid_score line the two up.
    """
    return {mid: idx for idx, mid in enumerate(movies_df["movieId"])}


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #

def hybrid_score(cb_scores: np.ndarray, cf_scores: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Weighted combination of aligned CB and CF score arrays."""
    return alpha * cb_scores + (1 - alpha) * cf_scores


def _align_cf_scores(cf_row: pd.Series, movies_df: pd.DataFrame, id_to_index: Dict[int, int]) -> np.ndarray:
    """Reindex a CF similarity row (indexed by movieId) into CB row order."""
    return np.array([cf_row[mid] if mid in cf_row.index else 0 for mid in movies_df["movieId"]])


# --------------------------------------------------------------------------- #
# Recommend by title
# --------------------------------------------------------------------------- #

def hybrid_recommend(
    title: str,
    movies_df: pd.DataFrame,
    cb_similarity: np.ndarray,
    item_similarity_cf: pd.DataFrame,
    id_to_index: Dict[int, int],
    top_n: int = 5,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """Recommend the top_n movies most similar to `title`, blending CB and CF."""
    matches = movies_df[movies_df["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        print(f"Title '{title}' not found.")
        return None

    if len(matches) > 1:
        print(f"Warning: {len(matches)} movies match '{title}'. Using: {matches['title'].iloc[0]}")

    movie_idx = matches.index[0]
    movie_id = matches["movieId"].iloc[0]

    cb_scores = cb_similarity[movie_idx]
    cf_scores_aligned = _align_cf_scores(item_similarity_cf.loc[movie_id], movies_df, id_to_index)

    scores = hybrid_score(cb_scores, cf_scores_aligned, alpha)
    indices = scores.argsort()[::-1][1 : top_n + 1]  # skip index 0: the movie itself

    display_cols = [c for c in ["title", "genres", "description"] if c in movies_df.columns]
    return movies_df.iloc[indices][display_cols]


# --------------------------------------------------------------------------- #
# Recommend for a user (used by the evaluation below)
# --------------------------------------------------------------------------- #

def recommend_for_user(
    user_id: int,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    cb_similarity: np.ndarray,
    item_similarity_cf: pd.DataFrame,
    id_to_index: Dict[int, int],
    top_n: int = 10,
    alpha: float = 0.7,
    like_threshold: float = 4,
) -> List[int]:
    """Recommend top_n unseen movieIds for a user, based on their liked movies."""
    liked_movies = train_df[
        (train_df["userId"] == user_id) & (train_df["rating"] >= like_threshold)
    ]["movieId"].tolist()
    liked_movies = [m for m in liked_movies if m in id_to_index]
    if not liked_movies:
        return []

    liked_idxs = [id_to_index[m] for m in liked_movies]
    cb_scores = cb_similarity[liked_idxs].mean(axis=0)

    cf_rows = [item_similarity_cf.loc[m] for m in liked_movies if m in item_similarity_cf.index]
    if cf_rows:
        cf_raw = pd.concat(cf_rows, axis=1).mean(axis=1)
        cf_scores = _align_cf_scores(cf_raw, movies_df, id_to_index)
    else:
        cf_scores = np.zeros_like(cb_scores)

    scores = hybrid_score(cb_scores, cf_scores, alpha)

    seen = set(liked_movies)
    ranked = scores.argsort()[::-1]

    recs = []
    for idx in ranked:
        mid = movies_df.iloc[idx]["movieId"]
        if mid not in seen:
            recs.append(mid)
        if len(recs) == top_n:
            break
    return recs


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def precision_recall_at_k(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    cb_similarity: np.ndarray,
    item_similarity_cf: pd.DataFrame,
    id_to_index: Dict[int, int],
    k: int = 10,
    alpha: float = 0.7,
) -> tuple:
    """Average Precision@k and Recall@k across all users in test_df."""
    precisions, recalls = [], []

    for user_id, group in test_df.groupby("userId"):
        relevant_items = set(group["movieId"])
        recommended_items = set(
            recommend_for_user(
                user_id, train_df, movies_df, cb_similarity, item_similarity_cf, id_to_index,
                top_n=k, alpha=alpha,
            )
        )

        n_hits = len(relevant_items & recommended_items)
        precisions.append(n_hits / k)
        recalls.append(n_hits / len(relevant_items))

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def evaluate_alphas(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    cb_similarity: np.ndarray,
    item_similarity_cf: pd.DataFrame,
    id_to_index: Dict[int, int],
    alphas: List[float] = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0),
    k: int = 10,
) -> tuple:
    """Run precision_recall_at_k across several alpha values, for the
    precision/recall-vs-alpha plot. Returns (results_precision, results_recall),
    both dicts keyed by alpha.
    """
    results_precision, results_recall = {}, {}
    for a in alphas:
        p, r = precision_recall_at_k(
            test_df, train_df, movies_df, cb_similarity, item_similarity_cf, id_to_index,
            k=k, alpha=a,
        )
        results_precision[a] = p
        results_recall[a] = r
        print(f"alpha={a}: Precision@{k} = {p:.4f}, Recall@{k} = {r:.4f}")
    return results_precision, results_recall
