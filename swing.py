
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import get_ndcg, get_precision_recall


def data_process(data_path):
    rating_data = pd.read_csv(data_path, sep=',', usecols=['user_id', 'movie_id', 'rating'],
                              dtype={'user_id': str, 'movie_id': str}).drop_duplicates().dropna()
    train, test = train_test_split(rating_data, test_size=0.2)
    return rating_data, test


if __name__ == '__main__':
    data_path = './movielens_sample.txt'
    train_data, test_data = data_process(data_path)
    sparse_features = ['user_id', 'movie_id']

    for feat in sparse_features:
        lbe = LabelEncoder()
        train_data[feat] = lbe.fit_transform(train_data[feat])
        test_data[feat] = lbe.transform(test_data[feat])

    # train_data = train_data[train_data['rating'] > 3]

    user_col, item_col, label_col = 'user_id', 'movie_id', 'rating'
    item_set = train_data[item_col].unique()
    user_set = train_data[user_col].unique()
    item_users_dict = train_data[[user_col, item_col]].groupby(item_col).agg(list).to_dict()[user_col]
    user_items_dict = train_data[[user_col, item_col]].groupby(user_col).agg(list).to_dict()[item_col]

    n_item = len(item_set)
    n_user = len(user_set)
    sim_mat = np.zeros((n_item, n_item))

    alpha = 1

    for i_item in range(1, len(item_set)):
        score = 0
        pre_item = item_set[i_item - 1]
        post_item = item_set[i_item]
        pre_item_users = item_users_dict[pre_item]
        post_item_users = item_users_dict[post_item]
        for pre_user in pre_item_users:
            for post_user in post_item_users:
                pre_user_item = set(user_items_dict[pre_user])
                post_user_item = set(user_items_dict[post_user])
                inter_num = len(pre_user_item.intersection(post_user_item))
                score += 1/(alpha + inter_num)
        sim_mat[pre_item, post_item] = score

    score_mat = np.zeros((n_user, n_item))
    for i in train_data.itertuples():
        score_mat[getattr(i, user_col), getattr(i, item_col)] = getattr(i, label_col)

    cut_num = 10
    pred = score_mat.dot(sim_mat)
    mask = np.where(score_mat == 0, 1, 0)
    pred = pred*mask
    preds = np.argsort(-pred, axis=1)[:, :cut_num]

    test_data['labels'] = test_data[item_col].map(str) + ':' + test_data[label_col].map(str)
    data_group = test_data[[user_col, 'labels']].groupby(user_col).agg(list).reset_index()
    def rank(row):
        new_row = []
        for item in row:
            pair = (int(item.split(':')[0]), float(item.split(':')[1]))
            new_row.append(pair)
        sort_row = sorted(new_row, key=lambda x: x[1], reverse=True)
        sort_row = list(map(lambda x: x[0], sort_row))
        res = np.array(sort_row)
        return res
    data_group['labels'] = data_group['labels'].apply(rank)
    labels = data_group.set_index(user_col).to_dict()['labels']
    precision, recall, ndcg = 0, 0, 0
    for i, label in labels.items():
        tmp_pre, tmp_recall = get_precision_recall(label, preds[i])
        tmp_ndcg = get_ndcg(label, preds[i], top_k=cut_num)
        precision += tmp_pre
        recall += tmp_recall
        ndcg += tmp_ndcg
    print("precision: {:.3f}, recall: {:.3f}, ndcg: {:.3f}".format(precision/len(labels), recall/len(labels), ndcg/len(labels)))


