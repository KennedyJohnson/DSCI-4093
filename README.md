# DSCI 4093 — Movie Recommender System

Senior capstone project analyzing the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/32m/) (~32 million ratings, ~87,000 movies) to build and evaluate collaborative-filtering movie recommenders.

## What it does

- **Exploratory analysis** (`CountMoviesByGenre.ipynb`, `FrequencyOfRatings.ipynb`): cleans the raw MovieLens tables and visualizes movie counts by genre and the distribution of user ratings.
- **Recommender modeling** (`DSCI4093_MovieRecommender.ipynb`, Python/[LensKit](https://lkpy.readthedocs.io/)): samples users with 100+ ratings, runs 5-fold cross-validation, and trains/evaluates **User-User** and **Item-Item** collaborative filtering models on RMSE. Also surfaces the most-rated and highest-rated movies overall and by genre (e.g. top comedies with 10,000+ reviews).
- **Algorithm comparison** (`Rcode/movieLens.Rmd`, R/[recommenderlab](https://cran.r-project.org/package=recommenderlab)): builds rating matrices at three sample sizes (`subsetData/ratings_small.csv`, `_medium.csv`, `_large.csv`), then cross-validates and compares 8 recommender algorithms — UBCF, IBCF, POPULAR, RANDOM, LIBMF, SVD, SVDF, and ALS — on RMSE/MSE/MAE, plus generates ROC curves benchmarking user-based CF, item-based CF, popularity, and random baselines against each other.
- **Output** (`output graphs/`): saved evaluation plots from the R analysis (e.g. `recommenderLab1000samplesPopularRandomUser`).

## Data

`ml-32m/` contains the full MovieLens 32M dataset (movies, ratings, tags, links). `subsetData/` holds pre-sampled rating matrices used to keep the R cross-validation runs tractable across algorithms.

## Stack

Python (pandas, matplotlib, LensKit) and R (recommenderlab) for two independent, complementary takes on the same recommendation problem — Python for exploration and CF baselines, R for a broader algorithm bake-off with ROC evaluation.
