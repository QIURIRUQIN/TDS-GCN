import numpy as np
import torch.utils.data as data


class SeqDataset(data.Dataset):

    def __init__(self, data, max_len, n_items, use_rating = False, data_aug=False, is_train=True, num_neg_sample=49):
        super(SeqDataset, self).__init__()
        self.data = data                     # data with neg samples
        self.max_len = max_len               # sequence max length
        self.num_neg_sample = num_neg_sample # total neg sample needed
        self.n_items = n_items
        self.use_rating = use_rating
        self.data_aug = data_aug
        self.is_train = is_train
        
        if data_aug and is_train:
           self._data_augment()

    def _data_augment(self):
        
        aug_data = []
        total_num = len(self.data)
        
        for idx in range(total_num):
            inter = self.data[idx]
            
            if len(inter['sequence']) <=3:
                continue
            
            num_sample_to_add = len(inter['sequence'])-3
            user_idx = inter['user_idx']
            raw_seq = inter['sequence']
            rating = inter['rating']
            
            
            for pos in range(num_sample_to_add):
                sub_seq = raw_seq[:3+pos]
                sub_rating = rating[:3+pos]
                new_pos_target = raw_seq[3+pos]
                new_neg_target = self._gen_negative_samples(sub_seq, new_pos_target)
                
                aug_data.append({'user_idx': user_idx,
                                 'sequence': sub_seq,
                                 'rating': sub_rating,
                                 'pos_target': new_pos_target,
                                 'neg_target': new_neg_target
                                 })
        
        print(f"Generated {len(aug_data)} sub-sequence data for training.")
        self.data.extend(aug_data)
    
    def _process_sequence(self, sequence):

        if len(sequence) > self.max_len:
            return sequence[-self.max_len:]  # keel max_len's latest interactions
        elif len(sequence) < self.max_len:
            return sequence + [self.n_items] * (self.max_len - len(sequence)) # padding

        return sequence

    def _gen_negative_samples(self, seq, pos_target, neg_target = None):
        if neg_target:
           num_to_sample = self.num_neg_sample - len(neg_target)
           if num_to_sample <=0:
               raise ValueError('No extra negative samples needed')
        else:
            num_to_sample = self.num_neg_sample

        neg_samples_to_add = []
        pos_all = seq.copy()
        pos_all.append(pos_target)
        while len(neg_samples_to_add) <= num_to_sample:
            idx = np.random.randint(0, self.n_items)
            if idx not in pos_all:
                neg_samples_to_add.append(idx)

        if neg_target:
           return neg_target.extend(neg_samples_to_add)
        else:
           return neg_samples_to_add

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inter = self.data[index]
        user_idx = inter['user_idx']
        raw_seq = inter['sequence']
        rating = inter['rating']
        pos_target = inter['pos_target']
        neg_target = inter['neg_target']

        processed_seq = self._process_sequence(raw_seq)
        processed_seq = np.array(processed_seq, dtype=np.int64)
        processed_rating = self._process_sequence(rating)
        processed_rating = np.array(processed_rating, dtype=np.int64)

        # raw_seq_len = len(raw_sequence)
        # positions = self._process_positions(raw_seq_len)
        # positions = np.array(positions, dtype=np.int64)
        neg_target = np.array(neg_target, dtype=np.int64)

        if self.use_rating:
            return user_idx, processed_seq, processed_rating, pos_target, neg_target
        else:
            return processed_seq, pos_target, neg_target

