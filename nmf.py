from surprise import Dataset, Reader, NMF
from collections import defaultdict
import pandas as pd

def nmf_model(train_df, test_df, N=10, n_factors=15, n_epochs=10, reg_pu=0.06):
    train_data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], Reader(rating_scale=(0, 5)))
    trainset = train_data.build_full_trainset()

    algo = NMF(n_factors=n_factors, n_epochs=n_epochs, reg_pu=reg_pu)
    algo.fit(trainset)

    all_items = set(train_df["item_id"].unique())
    rec_dict = defaultdict(list)

    for uid_inner in trainset.all_users():
        uid_raw = trainset.to_raw_uid(uid_inner)
        seen_items = set([trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[uid_inner]])
        unseen_items = all_items - seen_items

        predictions = [algo.predict(uid_raw, iid) for iid in unseen_items]
        top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:N]

        for pred in top_preds:
            rec_dict[uid_raw].append((pred.iid, pred.est))

    rec_list = []
    for uid, recs in rec_dict.items():
        for iid, score in recs:
            rec_list.append({"user_id": uid, "item_id": iid, "pred_rating": score})

    return pd.DataFrame(rec_list)

if __name__ == "__main__":
    print("This module is not meant to be run standalone.")
