from sklearn.metrics import mean_squared_error
import numpy as np

def mean_reciprocal_rank(y_true, y_pred):
    rr_total = 0
    for user_id in y_true['user_id'].unique():
        true_items = y_true[y_true['user_id'] == user_id]['item_id'].tolist()
        pred_items = y_pred[y_pred['user_id'] == user_id].sort_values(by='pred_rating', ascending=False)['item_id'].tolist()
        for rank, item in enumerate(pred_items):
            if item in true_items:
                rr_total += 1 / (rank + 1)
                break
    return rr_total / len(y_true['user_id'].unique())

def root_mean_squared_error(y_true, y_pred):
    merged = y_true.merge(y_pred, on=['user_id', 'item_id'], how='inner')
    return np.sqrt(mean_squared_error(merged['rating'], merged['pred_rating']))

def evaluate_all(results_dict, test_df):
    print("\n===== Evaluation Results =====")
    for name, pred_df in results_dict.items():
        mrr = mean_reciprocal_rank(test_df, pred_df)
        rmse = root_mean_squared_error(test_df, pred_df)
        print(f"{name}\n  MRR:  {mrr:.4f}\n  RMSE: {rmse:.4f}")
