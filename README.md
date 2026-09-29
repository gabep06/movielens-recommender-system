# Movie Recommender System with Matrix Factorization (SVD)

Builds a collaborative filtering recommender using Singular Value Decomposition (SVD), and tests whether it actually beats a simple popularity baseline.

## Goal

Build a movie recommender using SVD-based collaborative filtering, and measure whether the added complexity is worth it compared to just recommending what's popular.

## Approach

SVD-based collaborative filtering assumes user ratings can be explained by a small number of hidden factors (think: genre taste, mood, era preference) that aren't directly labeled in the data. By decomposing the user-movie rating matrix into `U`, `sigma`, and `Vt`, we reconstruct a dense prediction matrix that fills in ratings for movies a user hasn't seen yet, based on patterns learned from everyone else.

## Dataset

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) — 610 users, 9,724 movies, 100,836 ratings. After filtering out movies with fewer than 10 ratings: 610 users, 2,269 movies, 81,116 ratings.

The raw data is **not included** in this repo. Download `ml-latest-small.zip` from the link above and unzip it into a folder named `ml-latest-small` next to the notebook.

## Methods

- **Preprocessing:** Filtered out movies with fewer than 10 ratings. Split ratings per user (80% train / 20% test) so every user appears in both sets.
- **Mean-centering:** Subtracted each user's average rating before decomposition, to prevent generous/harsh raters from skewing the hidden factors.
- **Matrix factorization:** Decomposed the centered rating matrix with `scipy.sparse.linalg.svds` using `k=10` hidden factors, then reconstructed a dense prediction matrix.
- **Evaluation:** RMSE on held-out ratings, plus Precision@10 and Recall@10 compared against a popularity baseline.

## Results

**Sparsity:** 98.30% of the user-movie matrix is empty — most users only rated a small fraction of the 9,724 available movies.

**RMSE:** 0.8935 stars (predictions are typically within about 0.89 stars of a user's actual rating).

| Model | Precision@10 | Recall@10 |
|---|---|---|
| Popularity | **0.1306** | **0.1155** |
| SVD Collaborative Filtering | 0.1293 | 0.1056 |

## Key Findings

SVD did not beat the popularity baseline. This is likely because most users only rated a small number of movies, so SVD didn't have enough data to learn strong individual preferences, while the popularity baseline doesn't rely on individual data at all — it just recommends what's broadly popular. This is a useful finding: it shows that a more sophisticated technique isn't automatically better, and that testing against a simple baseline matters before assuming a complex model is worth using.

Another limitation: SVD only learns from rating patterns and never sees genre, plot, or other information about the movie itself — signals a content-based model could use directly.

## Tech Stack

- Python (pandas, NumPy, SciPy, Matplotlib)
- Jupyter Notebook

## Repository Structure

```
movielens-recommender-system/
├── README.md
├── svd_movie_recommender.ipynb
└── ml-latest-small/     # not included — download separately, see Dataset section
```

## Running This Project

```bash
git clone https://github.com/gabep06/movielens-recommender-system.git
cd movielens-recommender-system
pip install pandas numpy scipy matplotlib jupyter
jupyter notebook svd_movie_recommender.ipynb
```

## Future Work

- Tune `k` (try 5, 20, 50) and see how RMSE and Precision@10 change.
- Add the content-based genre model from my other project as a third point of comparison.
- Build a simple hybrid that averages SVD and content-based scores.

## Author

Gabriel Pereira — B.S. Data Science, University of Tampa
[LinkedIn](http://www.linkedin.com/in/gabriel-pereira-05b593359) · [GitHub](https://github.com/gabep06)
