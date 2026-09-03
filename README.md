# Hybrid Movie Recommendation System

A hybrid movie recommender built on the [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small` dataset (9,742 movies, 100,836 ratings), combining:

- **Content-Based Filtering (CB)** — sentence embeddings over each movie's description (fetched from TMDB), genres, tags and title.
- **Collaborative Filtering (CF)** — item-item similarity from user rating patterns.

The two are blended with a single weight `alpha`:

```
hybrid_score = alpha * CB_score + (1 - alpha) * CF_score
```

`alpha = 1.0` → pure content-based · `alpha = 0.0` → pure collaborative.

## Why hybrid?

This project isn't about picking a winner between CB and CF and discarding the other — `alpha` is a tunable parameter, not a binary choice, and each approach covers the other's blind spots:

- **CF** captures real personal taste well when there's enough rating history — but has **zero signal** for a movie that was just added (no one has rated it yet) or a user who just signed up (no ratings at all). This is the classic **cold-start problem**.
- **CB** works from day one for both cases: a new movie still has a description/genres/tags to embed, and a new user's stated preferences (or even just the genres they browse) are enough to get started. It also tends to add thematic diversity that pure CF, biased toward "what similar users already liked," can miss.

The Precision@10/Recall@10 evaluation below answers a narrower question — *for users with dense rating history, which alpha maximizes exact top-k hits?* — and the answer (a lower alpha) is useful for calibrating the system, not a reason to drop the content-based half. A production system would realistically vary `alpha` by context: higher for new users/movies, lower for users with rich history.

## Project structure

```
HybridRecommendationSystem/
│
├── data/                        # datasets (.csv) — not committed, see .gitignore
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── movies_with_descriptions.csv   # generated on first run (TMDB cache)
│
├── models/                      # cached embeddings
│   └── movie_embeddings.pkl     # generated on first run
│
├── notebooks/
│   └── Recommendation_Hybrid.ipynb    # orchestrates src/, visualizes results
│
├── src/                         # modular pipeline code
│   ├── __init__.py
│   ├── preprocessing.py         # load, clean, merge, TMDB enrichment, train/test split
│   ├── content_based.py         # TF-IDF baseline + embedding-based content model
│   ├── collaborative.py         # item-item collaborative filtering
│   └── hybrid.py                # score blending, recommend, evaluation
│
├── results/                     # exported plots/metrics (optional)
├── .env                         # TMDB_API_KEY — never committed
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and `cd` into it.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Get a free [TMDB API key](https://www.themoviedb.org/settings/api) and create a `.env` file in the project root:
   ```
   TMDB_API_KEY=your_key_here
   ```
5. Place the MovieLens `ml-latest-small` CSVs (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`) in `data/`.
6. Launch Jupyter **from the project root** (not from inside `notebooks/`, or the `src` imports and `data/` paths will break):
   ```bash
   jupyter notebook
   ```
7. Open `notebooks/Recommendation_Hybrid.ipynb` and run all cells. The first run is slower (TMDB fetch + embedding computation); both are cached to disk afterward.

## Usage

```python
from src.preprocessing import clean_data, multi_holdout_split
from src.content_based import build_content_model
from src.collaborative import build_collaborative_model
from src.hybrid import build_id_index, hybrid_recommend

ratings, movies = clean_data(data_dir="data")
ratings_train, ratings_test = multi_holdout_split(ratings)

embeddings, cb_similarity = build_content_model(movies)
item_similarity_cf = build_collaborative_model(ratings_train)
id_to_index = build_id_index(movies)

hybrid_recommend("Matrix", movies, cb_similarity, item_similarity_cf, id_to_index, alpha=0.7)
```

## Evaluation

The notebook evaluates Precision@10 and Recall@10 across several `alpha` values, using a per-user holdout split (`multi_holdout_split` hides a few movies each user rated highly, then checks whether the model recovers them). On this dataset, both metrics are highest at **alpha = 0.0** (pure collaborative filtering) and decrease steadily as more weight shifts to content-based.

**Why CF wins on this specific test:** CF is built from exactly the kind of signal this metric rewards — "users similar to you liked these specific movies." Content-based similarity, by contrast, measures *thematic* closeness (genre/description/tags), which doesn't necessarily track *personal taste*: two movies can be thematically close without the same person loving both. With ~165 ratings/user on average, this dataset is dense enough for CF's user-similarity signal to be strong.

**A caveat:** part of this gap may be structural to the evaluation method itself (CF's signal is inherently closer to what a "recover the exact hidden movie" metric rewards), rather than proof that the embedding model is a weak content representation. Content-based still matters where CF has no signal at all — cold-start for new users/movies with no ratings yet — and for thematic diversity/explainability in recommendations. The absolute metric values (~2-3% Precision@10) are also expected given the task difficulty: recovering 3 specific hidden movies per user out of ~9,700 candidates in a top-10 list — the relative comparison across alpha values matters more than the absolute magnitude.

## Tech stack

Python · pandas · scikit-learn (TF-IDF, cosine similarity) · sentence-transformers (`all-mpnet-base-v2`) · TMDB API · matplotlib
