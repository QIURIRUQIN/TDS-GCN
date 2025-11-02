import numpy as np
import torch.utils.data as data
from tqdm import tqdm

class MyDataset(data.Dataset):
    def __init__(self, data, n_items, train_mat, n_NegSamples, is_training):
        super(MyDataset, self).__init__()

        self.data = np.array(data)
        self.n_items = n_items
        self.train_mat = train_mat
        self.n_NegSample = n_NegSamples
        self.is_training = is_training

    def sample_ng(self):
        assert self.is_training
        tmp_trainMat = self.train_mat.todok()
        length = self.data.shape[0]
        self.neg_data = np.random.randint(low=0, high=self.n_items, size=length)

        for i in range(length):
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in tmp_trainMat:
                while (uid, iid) in tmp_trainMat:
                    iid = np.random.randint(low=0, high=self.n_items)
            self.neg_data[i] = iid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_idx = self.data[index][0]
        item_idx = self.data[index][1]
        if self.is_training:
            neg_data = self.neg_data
            item_j = neg_data[index]
            return user_idx, item_idx, item_j
        else:
            return user_idx, item_idx
