import pandas as pd
import numpy as np
import os
import json
import sys
import pickle
from tqdm import tqdm


file_path = os.path.dirname(sys.argv[0])
data_path = file_path.replace('data_process', 'data')

path = {
    'data_input': data_path + '/yelp/raw/',
    'dataset': data_path + '/yelp/dataset/'
}

os.makedirs(path['data_input'], exist_ok=True)
os.makedirs(path['dataset'], exist_ok=True)


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

def data_preparation_4seq(df: pd.DataFrame = None):
    if df is None:
        raise ValueError(f'df can not be None, please check!')

    df = df.sort_values(by=['user_idx', 'date']).reset_index(drop=True)
    train_data, val_data, test_data = [], [], []

    for user, group in df.groupby('user_idx'):
        inter = group['item_idx'].to_list()
        rating = group['rating'].to_list()
        n = len(inter)

        if n < 5:
            print(f"the number of interactions of {user} is too limited (need at least 5), current: {n}")
            continue

        # train: (:n-3) -> n-2
        # valid: (:n-2) -> n-1
        # test:  (:n-1) -> n
        train_seq, train_rating = inter[:n-3], rating[:n-3]
        train_target = inter[n-3]

        val_seq, val_raing = inter[:n-2], rating[:n-2]
        val_target = inter[n-2]

        test_seq, test_rating = inter[:n-1], rating[:n-1]
        test_target = inter[n-1]

        train_data.append({'user_idx': user, 'sequence': train_seq, 'rating': train_rating,'pos_target': train_target})
        val_data.append({'user_idx': user, 'sequence': val_seq, 'rating': val_raing, 'pos_target': val_target})
        test_data.append({'user_idx': user, 'sequence': test_seq, 'rating': test_rating,'pos_target': test_target})

    return train_data, val_data, test_data

def iterative_filer(df: pd.DataFrame = None, user_col: str = 'user_id',
                    item_col: str = 'business_id', min_user_inter=5, min_item_inter=5):
    if df is None:
        df = pd.read_excel(path['data_input'] + 'yelp_academic_dataset_business.xlsx')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.loc[(df['date'] >= '2018-01-01') & (df['date'] <= '2022-01-01')]
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

def genNegSample_4Seq(data,  n_items: int, n_negSamples: int=50):

    user_pos_items = {}

    for info in data:
        user_id = info['user_idx']
        pos_items = set(info['sequence']) | {info['pos_target']}
        if user_id in user_pos_items:
            user_pos_items[user_id].update(pos_items)
        else:
            user_pos_items[user_id] = pos_items

    data_with_neg = []
    for info in data:
        user_id = info['user_idx']
        pos_set = user_pos_items[user_id]
        neg_targets = []

        while len(neg_targets) <= n_negSamples:
            neg_candidate = np.random.randint(1, n_items+1)
            if neg_candidate not in pos_set:
                neg_targets.append(neg_candidate)

        processed_info = {
            'user_idx': info['user_idx'],
            'sequence': info['sequence'],
            'rating': info['rating'],
            'pos_target': info['pos_target'],
            'neg_target': neg_targets
        }

        data_with_neg.append(processed_info)

    return data_with_neg


def GetYelp(datafile):
    data_list = []
    lines = open(datafile).readlines()
    for line in tqdm(lines):
        review = json.loads(line.strip())
        data = {
            'user_id': review['user_id'],
            'business_id': review['business_id'],
            'rating': review['stars'],
            'date': review['date']
        }

        data_list.append(data)

    return pd.DataFrame(data_list)


def main(start_date: str = '2018-01-01', end_date: str = '2022-01-01',
         min_user_inter: int = 5, min_item_inter: int = 5,
         p_train: float = 0.8, p_val: float = 0.1):
    df = pd.read_excel(path['data_input'] + 'yelp_academic_dataset_review.xlsx')
    df = df.rename({'stars':'rating'}, axis=1)
    # df = GetYelp(path['data_input'] + 'yelp_academic_dataset_review_full.json')

    print("successful load raw data.")

    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[(df['date'] >= start_date) & (df['date'] < end_date)].reset_index(drop=True)

    # filter user and items with limited interations
    df = iterative_filer(df, min_user_inter=min_user_inter, min_item_inter=min_item_inter)
    user2id, item2id = construct_id_map_idx(df)

    # cunstruct user-idx and item-idx
    df['user_idx'] = df['user_id'].map(user2id)
    df['item_idx'] = df['business_id'].map(item2id)

    df = df.drop(columns=['user_id', 'business_id'])

    train_data, val_data, test_data = data_preparation_4seq(df)
    n_users, n_items = len(user2id), len(item2id)

    print(f'total {n_users} users')
    print(f'total {n_items} items')

    # gen negative samples
    train_data = genNegSample_4Seq(train_data, n_items=n_items, n_negSamples=49)
    val_data = genNegSample_4Seq(val_data, n_items=n_items, n_negSamples=49)
    test_data = genNegSample_4Seq(test_data, n_items=n_items, n_negSamples=49)

    with open(path['dataset'] + 'train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    with open(path['dataset'] + 'val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)

    with open(path['dataset'] + 'test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    main()

