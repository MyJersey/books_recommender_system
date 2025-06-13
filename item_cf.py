from surprise import Dataset, Reader, KNNBasic
from collections import defaultdict
import pandas as pd

def item_cf(train_df, test_df, N=10, sim_name="cosine", min_k=1):
    train_data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], Reader(rating_scale=(0, 5)))
    trainset = train_data.build_full_trainset()

    sim_options = {"name": sim_name, "user_based": False}
    algo = KNNBasic(sim_options=sim_options, k=N, min_k=min_k)
    algo.fit(trainset)

    all_items = set(train_df["item_id"].unique())
    rec_dict = defaultdict(list)

    for uid_inner in trainset.all_users():
        uid_raw = trainset.to_raw_uid(uid_inner)
        seen_items = set([trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[uid_inner]])
        unseen_items = all_items - seen_items

        predictions = [algo.predict(uid_raw, iid) for iid in unseen_items]
        predictions.sort(key=lambda x: x.est, reverse=True)

        for pred in predictions[:N]:
            rec_dict[uid_raw].append((pred.iid, pred.est))

    rec_list = []
    for uid, recs in rec_dict.items():
        for iid, score in recs:
            rec_list.append({"user_id": uid, "item_id": iid, "pred_rating": score})

    return pd.DataFrame(rec_list)
