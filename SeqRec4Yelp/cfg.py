# cfg.py
from dataclasses import dataclass

from dataclasses import dataclass

@dataclass
class ModelConfig:
    dataset: str
    model_use: str = "GatedGRU4Rec"
    weighted_method: str = "rating_avg" # weighting mechanism for item embedding
    item_embd_dim: int = 128
    user_embd_dim: int = 128
    hidden_dim: int = 64  # hidden dim for RNN Based Model
    layers: int = 4       # layers for RNN Based Model
    max_len: int = 20     # max lengths of a sequence, >max_len will be truncated, <max_len will be padded

    num_items: int = 0
    num_users: int = 0

    def __post_init__(self):
        dataset_mapping = {
            "s": (1922, 3498),      # 2018 ~ 2021, sampled data
            "m1": (53576, 102861),  # 2018 ~ 2021, full data
            "m2": (45320, 86814)    # 2016 ~ 2019, full data
        }

        if self.dataset not in dataset_mapping:
            raise ValueError(
                f"dataset not supported：{self.dataset}, possible datasets：{sorted(dataset_mapping.keys())}"
            )

        self.num_items, self.num_users = dataset_mapping[self.dataset]

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    K: int = 10               # evaluation metrics xxx@K
    num_neg: int = 49         # negative sample aligned with split_dataset.py
    n_epochs: int = 100
    early_stop: int = 20
    optimizer_use: str = "adamw"
    loss_fn: str = "bpr"

@dataclass
class LoaderConfig:
    use_rating: bool = True
    data_aug: bool = True
    batch_size: int = 256

@dataclass
class Config:
    model: ModelConfig = ModelConfig(dataset='s')
    train: TrainConfig = TrainConfig()
    loader: LoaderConfig = LoaderConfig()

configs = Config()