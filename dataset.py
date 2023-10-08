from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from eurodataloader import EuroSATDataset
import pandas as pd
from logging import ERROR, INFO
from flwr.common.logger import log

def load_datasets( 
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    val_ratio: float = 0.3,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    """Creates the dataloaders to be fed into the model."""
    datasets, testset = _partition_data(num_clients, iid, balance, seed)
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


def load_data() -> Tuple[Dataset, Dataset]:
    root = '/kaggle/input/eurosat/EuroSAT/2750'
    training_df = pd.read_csv('/kaggle/input/eurosat/training_merged_file.csv')
    test_df = pd.read_csv('/kaggle/input/eurosat/eurosat_test.csv')
    
    transforms_data = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
    ])

    #define dataset
    trainset = EuroSATDataset(train_df=training_df, train_dir=root, is_train=True, transform=transforms_data)
    testset = EuroSATDataset(train_df=test_df, train_dir=root, is_train=True, transform=transforms_data)

    return trainset, testset


def _partition_data(
    num_clients: int = 10,
    iid: Optional[bool] = False,
    balance: Optional[bool] = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:

    trainset, testset = load_data()
    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients
    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        log(INFO, f"Each client receives {partition_size} amount of data in a randomly iid fashion")
    else:
        if balance:
            trainset = _balance_classes(trainset, seed)
            partition_size = int(len(trainset) / num_clients)
        shard_size = int(partition_size / 2)
        idxs = trainset.targets.argsort()
        sorted_data = Subset(trainset, idxs)
        tmp = []
        for idx in range(num_clients * 2):
            tmp.append(
                Subset(sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1)))
            )
        idxs_list = torch.randperm(
            num_clients * 2, generator=torch.Generator().manual_seed(seed)
        )
        datasets = [
            ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
            for i in range(num_clients)
        ]
    return datasets, testset

def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in class_counts:
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled