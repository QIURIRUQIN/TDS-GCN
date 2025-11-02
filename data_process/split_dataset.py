import pandas as pd
import numpy as np
import os
import json
from scipy.sparse import csr_matrix
import pickle

from utils.get_data_tools import get_raw_data_from_file
from utils.process_data_tools import sampleHelper, generate_multi_label_adj
from data_process.generate_adj import generate_uu_adj, generate_ii_adj, generate_multi_adj, generate_multi_rating_label_adj
from CONST.path_const import root_path

data_path = os.path.join(root_path, 'data')
os.makedirs(data_path, exist_ok=True)

def construct_id_map_idx(df: pd.DataFrame = None, user_col: str = 'user_id', item_col: str = 'business_id'):
    if df is None:
        raise ValueError('df can not be None')

    user2id = {u: idx for idx, u in enumerate(np.unique(df[user_col]))}
    item2id = {i: idx for idx, i in enumerate(np.unique(df[item_col]))}

    with open(os.path.join(data_path, 'user2id.json'), 'w') as f:
        json.dump(user2id, f, indent=4)
    with open(os.path.join(data_path, 'item2id.json'), 'w') as f:
        json.dump(item2id, f, indent=4)

    return user2id, item2id

def data_preparation_V1(df: pd.DataFrame = None):
    if df is None:
        raise ValueError(f'df can not be None, please check!')
    
    df = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)
    train_list, val_list, test_list = [], [], []
    for user, group in df.groupby('user_id'):
        n = len(group)
        if n < 3:
            print(f"the number of interactions of {user} is too limited!")
            continue

        train_end = n - 2
        val_end = n - 1

        group_train = group.iloc[:train_end]
        group_val = group.iloc[train_end:val_end]
        group_test = group.iloc[val_end:]

        train_list.append(group_train)
        val_list.append(group_val)
        test_list.append(group_test)

    train_df = pd.concat(train_list, axis=0)
    val_df = pd.concat(val_list, axis=0)
    test_df = pd.concat(test_list, axis=0)
    
    return train_df, val_df, test_df

def iterative_filer(df: pd.DataFrame = None, user_col: str = 'user_id',
                    item_col: str = 'business_id', min_user_inter=5, min_item_inter=5):
    if df is None:
        df = get_raw_data_from_file(filename='yelp_academic_dataset_review', suffix='.xlsx')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.loc[(df['date'] >= '2021-01-01') & (df['date'] <= '2021-12-31')]
        df.reset_index(index=True, drop=True)

    prev_shape = None
    new_shape = df.shape
    while prev_shape != new_shape:
        prev_shape = new_shape
        # filter users
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= min_user_inter].index)].reset_index(drop=True)
        # filter items
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= min_item_inter].index)].reset_index(drop=True)

        new_shape = df.shape

    return df

def generate_ui_traj(df: pd.DataFrame, n_users: int, n_items: int, target_col: str = 'stars',
                     user_col: str = 'user_idx', item_col: str = 'item_idx'):

    return csr_matrix(
        (df[target_col], (df[user_col], df[item_col])),
        shape=(n_users, n_items)
    )

def genNegSample(n_negSamples: int = 10):
    with open(os.path.join(data_path, 'train_iter_class.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, 'val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(data_path, 'test.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    train = train_data.todok()
    test = test_data.todok()
    val = val_data.todok()
    test_u = test_data.tocoo().row
    test_i = test_data.tocoo().col
    val_u = val_data.tocoo().row
    val_i = val_data.tocoo().col
    assert len(test_u) == len(test_i)
    assert len(val_u) == len(val_i)
    n_items = test_data.shape[1]

    n1 = len(test_u)
    test_new = []
    for i in range(n1):
        u_idx = test_u[i]
        i_idx = test_i[i]
        test_new.append([u_idx, i_idx])
        for _ in range(n_negSamples):
            negItemidx = sampleHelper(
                train_data=train,
                target_data=test,
                user_idx=u_idx,
                n_items=n_items
            )
            test_new.append([u_idx, negItemidx])

    n2 = len(val_u)
    val_new = []
    for i in range(n2):
        u_idx = val_u[i]
        i_idx = val_i[i]
        val_new.append([u_idx, i_idx])
        for _ in range(n_negSamples):
            negItemidx = sampleHelper(
                train_data=train,
                target_data=val,
                user_idx=u_idx,
                n_items=n_items
            )
            val_new.append([u_idx, negItemidx])

    with open(os.path.join(data_path, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_new, f)
    with open(os.path.join(data_path, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_new, f)

def generate_pos_neg_adj(df: pd.DataFrame, n_users: int, n_items: int):
    df['pos_interation'] = 0
    df.loc[(df['stars'] >= 4) & (df['stars'] <= 5), 'pos_interaction'] = 1
    df['pos_inter_timestamp'] = 0
    df.loc[df['pos_interaction'] == 1, 'pos_inter_timestamp'] = df['date']
    pos_inter = generate_ui_traj(df=df, n_users=n_users, n_items=n_items, target_col='pos_interation')
    pos_inter_timestamp = generate_ui_traj(df=df, n_users=n_users, n_items=n_items, target_col='pos_inter_timestamp')
    df.drop(columns=['pos_interation', 'pos_inter_timestamp'], inplace=True)

    df['neg_interaction'] = 0
    df.loc[(df['stars'] >=1) & (df['stars'] <= 2), 'neg_interaction'] = 1
    df['neg_inter_timestamp'] = 0
    df.loc[df['neg_interaction'] == 1, 'neg_inter_timestamp'] = df['date']
    neg_inter = generate_ui_traj(df=df, n_users=n_users, n_items=n_items, target_col='neg_interaction')
    neg_inter_timestamp = generate_ui_traj(df=df, n_users=n_users, n_items=n_items, target_col='neg_inter_timestamp')
    df.drop(columns=['neg_interaction', 'neg_inter_timestamp'], inplace=True)

    return pos_inter, pos_inter_timestamp, neg_inter, neg_inter_timestamp

def main(start_date: str = '2018-01-01', end_date: str = '2022-01-01', 
         min_user_inter: int = 5, min_item_inter: int = 5):
    df = get_raw_data_from_file(filename='yelp_academic_dataset_review', suffix='.xlsx')

    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[(df['date'] >= start_date) & (df['date'] < end_date)].reset_index(drop=True)
    df['date'] = df['date'].astype(int) / 10**9 # ns -> s

    # filter user and items with limited interations
    df = iterative_filer(df, min_user_inter=min_user_inter, min_item_inter=min_item_inter)
    user2id, item2id = construct_id_map_idx(df)
    train_df, val_df, test_df = data_preparation_V1(df)
    # cunstruct user-idx and item-idx
    for d in [train_df, val_df, test_df]:
        d['user_idx'] = d['user_id'].map(user2id)
        d['item_idx'] = d['business_id'].map(item2id)

    n_users, n_items = len(user2id), len(item2id)
    # save (-1, 1, 0)
    train_df['p_n_class'] = 0
    train_df.loc[(train_df['stars'] >= 1) & (train_df['stars'] <= 2), 'p_n_class'] = -1
    train_df.loc[(train_df['stars'] >= 4) & (train_df['stars'] <= 5), 'p_n_class'] = 1
    R_train_1 = generate_ui_traj(train_df, n_users=n_users, n_items=n_items, target_col='p_n_class')
    train_df = train_df.drop(columns=['p_n_class'])
    # save (1, 0)
    train_df['iter_label'] = 0
    train_df.loc[(train_df['stars'] >= 4) & (train_df['stars'] <= 5), 'iter_label'] = 1
    R_train_2 = generate_ui_traj(train_df, n_users=n_users, n_items=n_items, target_col='iter_label')
    train_df = train_df.drop(columns=['iter_label'])
    # save (0, 1, 2, 3)
    train_df['rating_label'] = 0
    train_df.loc[(train_df['stars'] >= 1) & (train_df['stars'] <= 2), 'rating_label'] = 1
    train_df.loc[train_df['stars'] == 3, 'rating_label'] = 2
    train_df.loc[(train_df['stars'] >= 4) & (train_df['stars'] <= 5), 'rating_label'] = 3
    R_train_3 = generate_ui_traj(train_df, n_users=n_users, n_items=n_items, target_col='rating_label')
    # create multi-label adj
    # (1) interaction_class
    R_train_multi_iter = generate_multi_label_adj(train_df, n_users, n_items, target_col='rating_label')
    # (2) interaction_time
    R_train_multi_time = generate_multi_label_adj(train_df, n_users, n_items, target_col='date')
    train_df.drop(columns=['rating_label'], inplace=True)

    # save (0, 1, 2, 3, 4, 5)
    train_df['rating_label'] = 0
    train_df.loc[train_df['stars'] == 1, 'rating_label'] = 1
    train_df.loc[train_df['stars'] == 2, 'rating_label'] = 2
    train_df.loc[train_df['stars'] == 3, 'rating_label'] = 3
    train_df.loc[train_df['stars'] == 4, 'rating_label'] = 4
    train_df.loc[train_df['stars'] == 5, 'rating_label'] = 5
    R_train_4 = generate_ui_traj(train_df, n_users=n_users, n_items=n_items, target_col='rating_label')
    # create multi-label adj for KGCN
    # (1) interaction_class
    R_train_multi_iter_for_KGCN = generate_multi_label_adj(train_df, n_users, n_items, target_col='rating_label', n_lables=5)
    # (2) interaction_time
    R_train_multi_time_for_KGCN = generate_multi_label_adj(train_df, n_users, n_items, target_col='date', n_lables=5)

    R_train_time = generate_ui_traj(train_df, target_col='date', n_users=n_users, n_items=n_items)

    R_val = generate_ui_traj(val_df, n_users=n_users, n_items=n_items)
    R_test = generate_ui_traj(test_df, n_users=n_users, n_items=n_items)


    with open(os.path.join(data_path, 'train_p_n_class.pkl'), 'wb') as f:
        pickle.dump(R_train_1, f)
    with open(os.path.join(data_path, 'train_iter_class.pkl'), 'wb') as f:
        pickle.dump(R_train_2, f)
    with open(os.path.join(data_path, 'train_rating_label.pkl'), 'wb') as f:
        pickle.dump(R_train_3, f)
    with open(os.path.join(data_path, 'train_time.pkl'), 'wb') as f:
        pickle.dump(R_train_time, f)
    with open(os.path.join(data_path, 'train_multi_iter.pkl'), 'wb') as f:
        pickle.dump(R_train_multi_iter, f)
    with open(os.path.join(data_path, 'train_multi_time.pkl'), 'wb') as f:
        pickle.dump(R_train_multi_time, f)
    with open(os.path.join(data_path, 'val.pkl'), 'wb') as f:
        pickle.dump(R_val, f)
    with open(os.path.join(data_path, 'test.pkl'), 'wb') as f:
        pickle.dump(R_test, f)

    with open(os.path.join(data_path, 'train_rating_label_for_KGCN.pkl'), 'wb') as f:
        pickle.dump(R_train_4, f)
    with open(os.path.join(data_path, 'train_multi_iter_KGCN.pkl'), 'wb') as f:
        pickle.dump(R_train_multi_iter_for_KGCN, f)
    with open(os.path.join(data_path, 'train_multi_time_KGCN.pkl'), 'wb') as f:
        pickle.dump(R_train_multi_time_for_KGCN, f)

    pos_inter, pos_inter_timestamp, neg_inter, neg_inter_timestamp = generate_pos_neg_adj(df=train_df, n_users=n_users, n_items=n_items)
    with open(os.path.join(data_path, 'pos_inter.pkl'), 'wb') as f:
        pickle.dump(pos_inter, f)
    with open(os.path.join(data_path, 'pos_inter_timestamp.pkl'), 'wb') as f:
        pickle.dump(pos_inter_timestamp, f)
    with open(os.path.join(data_path, 'neg_inter.pkl'), 'wb') as f:
        pickle.dump(neg_inter, f)
    with open(os.path.join(data_path, 'neg_inter_timestamp.pkl'), 'wb') as f:
        pickle.dump(neg_inter_timestamp, f)

    print("The adjacency matrix has been established!")

if __name__ == "__main__":
    main()
    genNegSample(n_negSamples=50)
    generate_uu_adj()
    generate_ii_adj()
    generate_multi_adj()
    generate_multi_rating_label_adj()
