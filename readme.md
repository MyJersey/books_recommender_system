# Book Recommendation Systems

This project implements and compares four book recommendation systems using the Amazon book rating dataset.  
It is part of the course **Data Science in Production** at **IT University of Copenhagen (ITU)**.

**Dataset**  
The dataset is publicly available on Kaggle:  
🔗 https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data  
It includes:
- `books_ratings_training.csv`: user-item ratings for training
- `books_ratings_test.csv`: user-item ratings for evaluation
- `books_meta.csv`: book metadata such as title, category, and price

---

## Models Implemented

Each model is implemented in its own Python file:

1. **`non_personalized.py`**  
   Recommends the most popular recent books based on confidence intervals.

2. **`content_based.py`**  
   Uses cosine similarity over metadata features and retrieves top-25 nearest neighbors.

3. **`item_cf.py`**  
   Item-Item collaborative filtering using user-item rating patterns.

4. **`nmf.py`**  
   Matrix factorization using Non-negative Matrix Factorization (NMF) via Surprise library.

---

## Evaluation

Each recommender returns a top-10 ranked list of books for each user.  
Two evaluation metrics are used (see `evaluation.py`):
- **MRR (Mean Reciprocal Rank)**: How early the relevant item appears in the list.
- **RMSE (Root Mean Square Error)**: Rating prediction accuracy.

---

## How to Run

1. Run the pipeline:
```bash
python main.py
```

2. Output:
- Two bar plots comparing MRR and RMSE across the four models.
- Example predictions.

---

