# Hybrid Movie Recommendation System

A hybrid movie recommender built on the [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small` dataset (9,742 movies, 100,836 ratings), combining:

- **Content-Based Filtering (CB)** — sentence embeddings over each movie's description (fetched from TMDB), genres, tags and title.
- **Collaborative Filtering (CF)** — item-item similarity from user rating patterns.

The two are blended with a single weight `alpha`:

```
hybrid_score = alpha * CB_score + (1 - alpha) * CF_score
```

`alpha = 1.0` → pure content-based · `alpha = 0.0` → pure collaborative.

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

The notebook evaluates Precision@10 and Recall@10 across several `alpha` values, using a per-user holdout split. On this dataset, both metrics peak around **alpha = 0.3** (CF-dominant), suggesting collaborative signal captures individual taste better than thematic similarity alone for this ranking task — though content-based still matters for cold-start (new users/movies) and thematic diversity.

## Tech stack

Python · pandas · scikit-learn (TF-IDF, cosine similarity) · sentence-transformers (`all-mpnet-base-v2`) · TMDB API · matplotlib
