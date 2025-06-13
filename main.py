import pandas as pd
from non_personalized import non_personalized
from content_based import content_based
from item_cf import item_cf
from nmf import nmf_model
from evaluation import evaluate_all

# Load data
train = pd.read_csv("amazon/books_ratings_training.csv")
test = pd.read_csv("amazon/books_ratings_test.csv")
meta = pd.read_csv("amazon/books_meta.csv")

# Generate recommendations
print("Running non-personalized recommender...")
rec_np = non_personalized(train, test, recent_days=90, topN=10)

print("Running content-based recommender...")
rec_cb = content_based(train, test, meta, k=25)

print("Running item-item collaborative filtering...")
rec_cf = item_cf(train, test)

print("Running NMF recommender...")
rec_nmf = nmf_model(train, test)

# Evaluate all
print("Evaluating models...")
results = {
    "Non-Personalized": rec_np,
    "Content-Based": rec_cb,
    "Item-CF": rec_cf,
    "NMF": rec_nmf
}
evaluate_all(results, test)
