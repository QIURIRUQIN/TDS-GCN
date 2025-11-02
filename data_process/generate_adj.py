import numpy as np
import pandas as pd
import os
import json
from scipy.sparse import csr_matrix
import pickle
from itertools import combinations
import scipy.sparse as sp

from CONST.path_const import root_path

def generate_uu_adj():
    user_info = pd.read_excel(os.path.join(root_path, 'data', 'yelp_academic_dataset_user.xlsx'))
    with open(os.path.join(root_path, 'data', 'user2id.json'), 'r', encoding='utf-8') as f:
        user2id = json.load(f)

    n = len(user2id)
    rows, cols = [], []
    for _, row in user_info.iterrows():
        if row['user_id'] not in user2id:
            continue

        u = row['user_id']
        u_idx = user2id[u]
        friends = row['friends']

        if pd.isna(friends) or not isinstance(friends, str):
            continue

        friends_list = [f.strip() for f in friends.split(',') if f.strip()]

        for friend in friends_list:
            if friend in user2id:
                friend_idx = user2id[friend]
                rows.extend([u_idx, friend_idx])
                cols.extend([friend_idx, u_idx])

    pairs = set(zip(rows, cols))
    rows, cols = zip(*pairs)

    adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    with open(os.path.join(root_path, 'data', 'uu_graph.pkl'), 'wb') as f:
        pickle.dump(adj_matrix, f)

def generate_ii_adj():
    business_info = pd.read_excel(os.path.join(root_path, 'data', 'yelp_academic_dataset_business.xlsx'))
    with open(os.path.join(root_path, 'data', 'item2id.json'), 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    business_info = business_info.loc[business_info['business_id'].isin(set(item2id.keys()))].reset_index(drop=True)

    n = len(item2id)
    rows, cols = [], []
    for city, group in business_info.groupby('city'):
        idx = [item2id[b] for b in group['business_id']]
        if len(idx) < 2:
            continue
        for i, j in combinations(idx, 2):
            rows.extend([i, j])
            cols.extend([j, i])

    pairs = set(zip(rows, cols))
    rows, cols = zip(*pairs)

    adj_matrix = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n)
    )

    with open(os.path.join(root_path, 'data', 'ii_graph.pkl'), 'wb') as f:
            pickle.dump(adj_matrix, f)
    
def generate_multi_adj():
    with open(os.path.join(root_path, 'data', 'train_p_n_class.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(root_path, 'data', 'train_time.pkl'), 'rb') as f:
        train_time = pickle.load(f)

    # construct [[0, R], [R^T, 0]]
    zero_u = sp.csr_matrix((train_data.shape[0], train_data.shape[0]))
    zero_i = sp.csr_matrix((train_data.shape[1], train_data.shape[1]))
    A = sp.bmat([
        [zero_u, train_data],
        [train_data.T, zero_i]
    ], format='csr')

    zero_u = sp.csr_matrix((train_time.shape[0], train_time.shape[0]))
    zero_i = sp.csr_matrix((train_time.shape[1], train_time.shape[1]))
    B = sp.bmat([
        [zero_u, train_time],
        [train_time.T, zero_i]
    ], format='csr')

    with open(os.path.join(root_path, 'data', 'multi_graph_A.pkl'), 'wb') as f:
        pickle.dump(A, f)
    with open(os.path.join(root_path, 'data', 'multi_graph_B.pkl'), 'wb') as f:
        pickle.dump(B, f)

    print("✅ Multi-adjacency matrices A and B have been generated and saved successfully!")

def generate_multi_rating_label_adj():
    with open(os.path.join(root_path, 'data', 'train_multi_iter.pkl'), 'rb') as f:
        train_multi_label_iter = pickle.load(f)
    with open(os.path.join(root_path, 'data', 'train_multi_time.pkl'), 'rb') as f:
        train_multi_label_time = pickle.load(f)

    # construct [[0, R], [R^T, 0]]
    zero_u = sp.csr_matrix((train_multi_label_iter.shape[0], train_multi_label_iter.shape[0]))
    zero_i = sp.csr_matrix((train_multi_label_iter.shape[1], train_multi_label_iter.shape[1]))
    A_multi_label = sp.bmat([
        [zero_u, train_multi_label_iter],
        [train_multi_label_iter.T, zero_i]
    ], format='csr')

    zero_u = sp.csr_matrix((train_multi_label_time.shape[0], train_multi_label_time.shape[0]))
    zero_i = sp.csr_matrix((train_multi_label_time.shape[1], train_multi_label_time.shape[1]))
    B_multi_label = sp.bmat([
        [zero_u, train_multi_label_time],
        [train_multi_label_time.T, zero_i]
    ], format='csr')

    with open(os.path.join(root_path, 'data', 'multi_label_graph_A.pkl'), 'wb') as f:
        pickle.dump(A_multi_label, f)
    with open(os.path.join(root_path, 'data', 'multi_label_graph_B.pkl'), 'wb') as f:
        pickle.dump(B_multi_label, f)

    print("✅ Multi-label-adjacency matrices A and B have been generated and saved successfully!")

def generate_multi_rating_label_adj_for_KGCN():
    with open(os.path.join(root_path, 'data', 'train_multi_iter_KGCN.pkl'), 'rb') as f:
        train_multi_label_iter = pickle.load(f)
    with open(os.path.join(root_path, 'data', 'train_multi_time_KGCN.pkl'), 'rb') as f:
        train_multi_label_time = pickle.load(f)

    # construct [[0, R], [R^T, 0]]
    zero_u = sp.csr_matrix((train_multi_label_iter.shape[0], train_multi_label_iter.shape[0]))
    zero_i = sp.csr_matrix((train_multi_label_iter.shape[1], train_multi_label_iter.shape[1]))
    A_multi_label = sp.bmat([
        [zero_u, train_multi_label_iter],
        [train_multi_label_iter.T, zero_i]
    ], format='csr')

    zero_u = sp.csr_matrix((train_multi_label_time.shape[0], train_multi_label_time.shape[0]))
    zero_i = sp.csr_matrix((train_multi_label_time.shape[1], train_multi_label_time.shape[1]))
    B_multi_label = sp.bmat([
        [zero_u, train_multi_label_time],
        [train_multi_label_time.T, zero_i]
    ], format='csr')

    with open(os.path.join(root_path, 'data', 'multi_label_graph_A_KGCN.pkl'), 'wb') as f:
        pickle.dump(A_multi_label, f)
    with open(os.path.join(root_path, 'data', 'multi_label_graph_B_KGCN.pkl'), 'wb') as f:
        pickle.dump(B_multi_label, f)

    print("✅ Multi-label-adjacency matrices A and B or KGCN have been generated and saved successfully!")

def generate_pos_neg_adj_for_TDSGCN():
    with open(os.path.join(root_path, 'data', 'pos_inter.pkl'), 'rb') as f:
        pos_inter = pickle.load(f)
    with open(os.path.join(root_path, 'data', 'pos_inter_timestamp.pkl'), 'rb') as f:
        pos_inter_timestamp = pickle.load(f)

    with open(os.path.join(root_path, 'data', 'neg_inter.pkl'), 'rb') as f:
        neg_inter = pickle.load(f)
    with open(os.path.join(root_path, 'data', 'neg_inter_timestamp.pkl'), 'rb') as f:
        neg_inter_timestamp = pickle.load(f)

    # construct [[0, R], [R^T, 0]]
    zero_u = sp.csr_matrix((pos_inter.shape[0], pos_inter.shape[0]))
    zero_i = sp.csr_matrix((pos_inter.shape[1], pos_inter.shape[1]))
    A_multi_label = sp.bmat([
        [zero_u, pos_inter],
        [pos_inter.T, zero_i]
    ], format='csr')

    zero_u = sp.csr_matrix((pos_inter_timestamp.shape[0], pos_inter_timestamp.shape[0]))
    zero_i = sp.csr_matrix((pos_inter_timestamp.shape[1], pos_inter_timestamp.shape[1]))
    B_multi_label = sp.bmat([
        [zero_u, pos_inter_timestamp],
        [pos_inter_timestamp.T, zero_i]
    ], format='csr')

    with open(os.path.join(root_path, 'data', 'pos_inter_TDSGCN.pkl'), 'wb') as f:
        pickle.dump(A_multi_label, f)
    with open(os.path.join(root_path, 'data', 'pos_inter_timestamp_TDSGCN.pkl'), 'wb') as f:
        pickle.dump(B_multi_label, f)

    del A_multi_label, B_multi_label

    zero_u = sp.csr_matrix((neg_inter.shape[0], neg_inter.shape[0]))
    zero_i = sp.csr_matrix((neg_inter.shape[1], neg_inter.shape[1]))
    A_multi_label = sp.bmat([
        [zero_u, neg_inter],
        [neg_inter.T, zero_i]
    ], format='csr')

    zero_u = sp.csr_matrix((neg_inter_timestamp.shape[0], neg_inter_timestamp.shape[0]))
    zero_i = sp.csr_matrix((neg_inter_timestamp.shape[1], neg_inter_timestamp.shape[1]))
    B_multi_label = sp.bmat([
        [zero_u, neg_inter_timestamp],
        [neg_inter_timestamp.T, zero_i]
    ], format='csr')

    with open(os.path.join(root_path, 'data', 'neg_inter_TDSGCN.pkl'), 'wb') as f:
        pickle.dump(A_multi_label, f)
    with open(os.path.join(root_path, 'data', 'neg_inter_timestamp_TDSGCN.pkl'), 'wb') as f:
        pickle.dump(B_multi_label, f)

    print("✅ Multi-label-adjacency matrices A and B or TDSGCN have been generated and saved successfully!")
