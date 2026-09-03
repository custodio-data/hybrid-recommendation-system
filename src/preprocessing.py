"""
preprocessing.py
-----------------
Data loading, cleaning, enrichment (TMDB descriptions), and train/test
splitting for the hybrid movie recommendation system.

All functions here are pure(ish): they take DataFrames/paths in and return
DataFrames out, so they can be unit-tested and re-used from the notebook
or from other scripts without side effects other than optional caching
to disk (TMDB descriptions, embeddings are handled in content_based.py).
"""

from pathlib import Path
from typing import Optional, Tuple

import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four MovieLens CSV files from `data_dir`.

    Returns
    -------
    ratings, movies, tags, links : pd.DataFrame
    """
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    links = pd.read_csv(data_dir / "links.csv")
    return ratings, movies, tags, links


# --------------------------------------------------------------------------- #
# Merging / cleaning
# --------------------------------------------------------------------------- #

def merge_movies_links(movies: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Merge movies with links (adds imdbId / tmdbId) on movieId."""
    return pd.merge(movies, links, on="movieId", how="left")


def clean_genres(movies_links: pd.DataFrame) -> pd.DataFrame:
    """Replace '|' with a space in genres so TF-IDF treats each genre as its own token."""
    movies_links = movies_links.copy()
    movies_links["genres_clean"] = movies_links["genres"].str.replace("|", " ", regex=False)
    return movies_links


def merge_tags(movies_links: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Merge movies with tags, collapsing all of a movie's tags into one string column."""
    merged = pd.merge(movies_links, tags, on="movieId", how="left")
    merged["tags"] = merged.groupby("movieId")["tag"].transform(
        lambda x: " ".join(x.dropna().unique())
    )
    merged = merged.drop_duplicates(subset="movieId").reset_index(drop=True)
    return merged


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Extract release year from the title (e.g. 'Toy Story (1995)') into its own column."""
    df = df.copy()
    df["year"] = df["title"].str.extract(r"\((\d{4})\)", expand=False).fillna("")
    return df


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Build the free-text field used for embeddings: description + genres + year + tags + title."""
    df = df.copy()
    df["combined_text"] = (
        df.get("description", pd.Series("", index=df.index)).fillna("") + " "
        + df["genres"].fillna("") + " "
        + df["year"].astype(str) + " "
        + df["tags"].fillna("") + " "
        + df["title"].fillna("")
    )
    return df


# --------------------------------------------------------------------------- #
# TMDB enrichment
# --------------------------------------------------------------------------- #

def get_tmdb_api_key() -> Optional[str]:
    """Load TMDB_API_KEY from a .env file (never hardcode it)."""
    load_dotenv()
    return os.getenv("TMDB_API_KEY")


def _fetch_description(tmdb_id, api_key: str, cache: dict) -> str:
    """Fetch a single movie's overview from TMDB, with an in-memory cache."""
    if tmdb_id in cache:
        return cache[tmdb_id]
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=en-US"
    response = requests.get(url, timeout=10)
    desc = ""
    if response.status_code == 200:
        desc = response.json().get("overview", "")
    cache[tmdb_id] = desc
    return desc


def fetch_tmdb_descriptions(
    movies_links: pd.DataFrame,
    api_key: str,
    cache_path: str = "data/movies_with_descriptions.csv",
    request_delay: float = 0.2,
) -> pd.DataFrame:
    """Add a `description` column fetched from TMDB, one call per movie.

    If `cache_path` already exists, loads from it instead of hitting the API
    again (TMDB descriptions don't change, and the free-tier rate limit
    makes re-fetching 9000+ movies wasteful).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"'{cache_path}' already exists — loading cached descriptions.")
        return pd.read_csv(cache_path)

    movies_links = movies_links.copy()
    movies_links["description"] = ""
    desc_cache: dict = {}

    for idx, tmdb_id in tqdm(
        zip(movies_links.index, movies_links["tmdbId"]),
        total=len(movies_links),
        desc="Fetching TMDB descriptions",
    ):
        if pd.notnull(tmdb_id):
            movies_links.at[idx, "description"] = _fetch_description(tmdb_id, api_key, desc_cache)
            time.sleep(request_delay)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    movies_links.to_csv(cache_path, index=False)
    print(f"Descriptions saved to '{cache_path}'.")
    return movies_links


# --------------------------------------------------------------------------- #
# Train / test split (for evaluating the recommender, not a generic ML split)
# --------------------------------------------------------------------------- #

def multi_holdout_split(
    ratings_df: pd.DataFrame,
    min_ratings: int = 8,
    like_threshold: float = 4,
    n_holdout: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user split: hold out `n_holdout` liked movies per user as the test set.

    Users with too few ratings or too few "liked" movies are kept entirely
    in the training set (nothing held out for them).
    """
    np.random.seed(random_state)
    test_rows, train_frames = [], []

    for uid, group in ratings_df.groupby("userId"):
        liked = group[group["rating"] >= like_threshold]
        if len(group) < min_ratings or len(liked) < n_holdout:
            train_frames.append(group)
            continue
        held_out = liked.sample(n_holdout, random_state=random_state)
        test_rows.append(held_out)
        train_frames.append(group.drop(held_out.index))

    ratings_train = pd.concat(train_frames).reset_index(drop=True)
    ratings_test = pd.concat(test_rows).reset_index(drop=True)
    return ratings_train, ratings_test


# --------------------------------------------------------------------------- #
# Orchestrator — this is what `from src.preprocessing import clean_data` calls
# --------------------------------------------------------------------------- #

def clean_data(
    data_dir: str = "data",
    fetch_descriptions: bool = True,
    tmdb_api_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline and return (ratings, movies_links_tags).

    `movies_links_tags` is the enriched, deduplicated movie table with
    `combined_text` ready to be embedded by `content_based.build_content_model`.
    """
    ratings, movies, tags, links = load_raw_data(data_dir)

    movies_links = merge_movies_links(movies, links)
    movies_links = clean_genres(movies_links)

    if fetch_descriptions:
        api_key = tmdb_api_key or get_tmdb_api_key()
        if not api_key:
            raise ValueError(
                "fetch_descriptions=True but no TMDB_API_KEY was found. "
                "Set it in your .env file or pass tmdb_api_key explicitly."
            )
        movies_links = fetch_tmdb_descriptions(
            movies_links, api_key, cache_path=f"{data_dir}/movies_with_descriptions.csv"
        )

    movies_links_tags = merge_tags(movies_links, tags)
    movies_links_tags = add_year_column(movies_links_tags)
    movies_links_tags = build_combined_text(movies_links_tags)

    return ratings, movies_links_tags
