import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

def sampleHelper(train_data: dict, target_data: dict, user_idx: int, n_items: int):
    j = np.random.randint(n_items)
    while (user_idx, j) in train_data or (user_idx, j) in target_data:
        j = np.random.randint(n_items)

    return j

def generate_multi_label_adj(train_df: pd.DataFrame, n_users: int, n_items: int, target_col: str = 'rating_label',
                     user_col: str = 'user_idx', item_col: str = 'item_idx', n_lables: int = 3):
    multi_adj_matrix = lil_matrix((n_users, n_lables * n_items), dtype=np.float32)

    for _, row in train_df.iterrows():
        u = row[user_col]
        i = row[item_col]
        offset = row['rating_label']
        col_idx = i * n_lables + (offset - 1)
        multi_adj_matrix[u, col_idx] = 1.0 if target_col == 'rating_label' else row[target_col]

    adj_matrix = multi_adj_matrix.tocsr()
    return adj_matrix
