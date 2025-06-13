def non_personalized(train, test, recent_days, topN, alpha=0.05):

    latest_time = train['timestamp'].max()
    recent_threshold = latest_time - recent_days * 24 * 60 * 60
    recent_train = train[train['timestamp'] >= recent_threshold]

    stats = recent_train.groupby('item_id')['rating'].agg(['mean', 'std', 'count'])
    stats['std'] = stats['std'].fillna(0)
    z = norm.ppf(1 - alpha / 2)
    stats['ci_lower'] = stats['mean'] - z * stats['std'] / np.sqrt(stats['count'])

    stats = stats.sort_values('ci_lower', ascending=False)
    top_items = stats.head(topN).index.tolist()

    all_users = test['user_id'].unique()
    item_mean_ratings = train.groupby('item_id')['rating'].mean()
    global_mean = train['rating'].mean()

    results = []
    for user in all_users:
        for item in top_items:
            pred_rating = item_mean_ratings.get(item, global_mean)
            results.append({'user_id': user, 'item_id': item, 'pred_rating': pred_rating})

    rec_df = pd.DataFrame(results)
    return rec_df

rec_df_np = non_personalized(train, test, recent_days=90, topN=10)
rec_df_np.head()