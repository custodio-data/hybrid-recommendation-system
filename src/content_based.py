"""
content_based.py
-----------------
Content-Based Filtering: recommends movies based on their own attributes
(genres, description, tags) rather than other users' behaviour.

Two approaches are included:
  1. TF-IDF over genres        -> quick baseline, no external dependencies.
  2. Sentence-embeddings over
     description+genres+tags   -> the richer model actually used by the
                                   hybrid system (build_content_model).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# Baseline: TF-IDF over genres
# --------------------------------------------------------------------------- #

def build_tfidf_genre_model(movies_df: pd.DataFrame) -> np.ndarray:
    """Build a genre-only TF-IDF cosine-similarity matrix.

    Expects a `genres_clean` column (see preprocessing.clean_genres).
    Kept as a lightweight baseline — the embedding model below is what
    the hybrid system actually uses.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres_clean"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


# --------------------------------------------------------------------------- #
# Embedding-based content model (the one used downstream by hybrid.py)
# --------------------------------------------------------------------------- #

def _safe_encode(text, model) -> np.ndarray:
    """Encode text with the sentence-transformer, falling back to a zero
    vector for empty/invalid text so the pipeline never crashes on NaNs."""
    if isinstance(text, str) and text.strip():
        return model.encode(text)
    return np.zeros(model.get_sentence_embedding_dimension())


def compute_embeddings(
    movies_df: pd.DataFrame,
    text_column: str = "combined_text",
    model_name: str = "all-mpnet-base-v2",
) -> np.ndarray:
    """Encode `text_column` for every row using a SentenceTransformer model.

    Returns a (n_movies, embedding_dim) array. This is the slow step
    (one forward pass per movie) — cache it with save_embeddings/
    load_embeddings whenever possible instead of recomputing.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = movies_df[text_column].apply(lambda t: _safe_encode(t, model))
    return np.vstack(embeddings.values)


def save_embeddings(embeddings: np.ndarray, path: str = "models/movie_embeddings.pkl") -> None:
    """Persist embeddings to a .pkl file so they don't need recomputing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(list(embeddings), f)
    print(f"Embeddings saved to '{path}'.")


def load_embeddings(path: str = "models/movie_embeddings.pkl") -> np.ndarray:
    """Load previously-computed embeddings from a .pkl file."""
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return np.vstack(embeddings)


def build_content_model(
    movies_df: pd.DataFrame,
    text_column: str = "combined_text",
    model_name: str = "all-mpnet-base-v2",
    embeddings_path: Optional[str] = "models/movie_embeddings.pkl",
    recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (or load) embeddings and the resulting cosine-similarity matrix.

    Parameters
    ----------
    movies_df : must contain `text_column` (see preprocessing.build_combined_text)
    embeddings_path : if given and the file exists, embeddings are loaded from
        it instead of recomputed — unless `recompute=True`.

    Returns
    -------
    embeddings : (n_movies, dim) array
    cb_similarity : (n_movies, n_movies) cosine-similarity matrix
    """
    if embeddings_path and Path(embeddings_path).exists() and not recompute:
        embeddings = load_embeddings(embeddings_path)
    else:
        embeddings = compute_embeddings(movies_df, text_column=text_column, model_name=model_name)
        if embeddings_path:
            save_embeddings(embeddings, embeddings_path)

    cb_similarity = cosine_similarity(embeddings)
    return embeddings, cb_similarity


# --------------------------------------------------------------------------- #
# Recommendation helpers
# --------------------------------------------------------------------------- #

def recommend_similar_movies(
    movie_title: str,
    movies_df: pd.DataFrame,
    cb_similarity: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """Exact-title lookup version (used with the TF-IDF genre model)."""
    idx = movies_df[movies_df["title"] == movie_title].index[0]
    sim_scores = list(enumerate(cb_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][["movieId", "title", "genres"]]


def recommend_content(
    title: str,
    movies_df: pd.DataFrame,
    cb_similarity: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """Flexible (substring, case-insensitive) title lookup, used with the
    embedding-based model. Warns if the title is ambiguous."""
    display_cols = [c for c in ["title", "description"] if c in movies_df.columns]
    matches = movies_df[movies_df["title"].str.contains(title, case=False, na=False)]

    if matches.empty:
        print(f"Title '{title}' not found.")
        return pd.DataFrame(columns=display_cols)

    if len(matches) > 1:
        print(f"Warning: {len(matches)} movies match '{title}'. Using: {matches['title'].iloc[0]}")

    idx = matches.index[0]
    sim_scores = list(enumerate(cb_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    recommendations = movies_df.iloc[[i for i, _ in sim_scores]][display_cols]
    return recommendations
