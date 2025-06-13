import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

def content_based(train, test, meta, k=25):
    # Merge metadata
    book_features = meta[['item_id', 'categories']].dropna()
    book_features['categories'] = book_features['categories'].apply(eval)

    mlb = MultiLabelBinarizer()
    category_matrix = mlb.fit_transform(book_features['categories'])

    item_ids = book_features['item_id'].values
    item_index = {item_id: i for i, item_id in enumerate(item_ids)}
    sim_matrix = cosine_similarity(category_matrix)

    user_ids = test['user_id'].unique()
    results = []

    train_avg = train.groupby('item_id')['rating'].mean().to_dict()
    global_avg = train['rating'].mean()

    for user in user_ids:
        user_train = train[train['user_id'] == user]
        seen_items = user_train['item_id'].tolist()
        rated_items = [(iid, train_avg.get(iid, global_avg)) for iid in seen_items if iid in item_index]

        scores = {}
        for iid, rating in rated_items:
            idx = item_index[iid]
            sim_scores = sim_matrix[idx]
            for j, sim in enumerate(sim_scores):
                rec_iid = item_ids[j]
                if rec_iid not in seen_items:
                    scores[rec_iid] = scores.get(rec_iid, 0) + sim * rating

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for iid, score in ranked_items:
            results.append({'user_id': user, 'item_id': iid, 'pred_rating': score})

    return pd.DataFrame(results)

