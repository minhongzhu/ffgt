from loader.grpe_collator import grpe_collator
from loader.vanilla_collator import vanilla_collator
from loader.wrapper import (
    MyZINCDataset,
    MyGraphPropPredDataset,
    MyPeptidesFunctionalDataset,
    MyPeptidesStructuralDataset,
    MyPatternDataset
)

import numpy as np
from numpy.random import default_rng

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
from functools import partial
import pdb

from loader.transform import pre_transform_in_memory
from loader.pos_enc import compute_LapPE
from loss.weighted_cross_entropy import weighted_cross_entropy


dataset = None

def get_dataset(dataset_name="abaaba"):
    global dataset
    if dataset is not None:
        return dataset

    if dataset_name == "ZINC":
        dataset = {
            "num_class": 1,
            "loss_fn": F.l1_loss,
            "metric": "mae",
            "metric_mode": "min",
            "train_dataset": MyZINCDataset(
                subset=True, root="dataset/pyg_zinc", split="train"
            ),
            "valid_dataset": MyZINCDataset(
                subset=True, root="dataset/pyg_zinc", split="val"
            ),
            "test_dataset": MyZINCDataset(
                subset=True, root="dataset/pyg_zinc", split="test"
            ),
            "max_node": 128,
        }
    elif dataset_name.startswith("PATTERN"):
        # datset_name: e.g. PATTERN_pq_0.1_0.005
        dataset = {
            "num_class": 1,
            "loss_fn": weighted_cross_entropy,
            "metric": "accuracy-SBM",
            "metric_mode": "max",
            "train_dataset": MyPatternDataset(
                root="dataset/SBM-PATTERN", name=dataset_name, split="train"
            ),
            "valid_dataset": MyPatternDataset(
                root="dataset/SBM-PATTERN", name=dataset_name, split="val"
            ),
            "test_dataset": MyPatternDataset(
                root="dataset/SBM-PATTERN", name=dataset_name, split="test"
            ),
            "max_node": 512
        }
    elif dataset_name == "pcba":
        dataset = {
            "num_class": 128,
            "loss_fn": F.binary_cross_entropy_with_logits,
            "metric": "ap",
            "metric_mode": "max",
            "dataset": MyGraphPropPredDataset("ogbg-molpcba", root="dataset"),
            "max_node": 512,
        }
    elif dataset_name == 'pep-func':
        dataset = {
            "num_class": 10,
            "loss_fn": F.binary_cross_entropy_with_logits,
            "metric": "ap",
            "metric_mode": "max",
            "dataset": MyPeptidesFunctionalDataset(root="dataset"),
            "max_node": 512
        }
    elif dataset_name == 'pep-struc':
        dataset = {
            "num_class": 11,
            "loss_fn": F.l1_loss,
            "metric": "mae",
            "metric_mode": "min",
            "dataset": MyPeptidesStructuralDataset(root='dataset'),
            "max_node": 512          
        }
    else:
        raise NotImplementedError

    print(f" > {dataset_name} loaded!")
    print(dataset)
    print(f" > dataset info ends")
    return dataset


class GraphDataModule:
    def __init__(
        self,
        dataset_name: str = "ZINC",
        task: str = "graph",
        model: str = 'grpe_ffgt',
        num_workers: int = 0,
        batch_size: int = 256,
        max_dist: int = 2,
        max_freq: int = 8,
        max_node: int = 512,
        subset: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model = model
        self.task = task
        self.num_vitural = 1 if task == 'graph' else 0
        self.one_hot_edge = False if self.dataset_name in ['voc-pixels'] else True

        self.dataset = get_dataset(self.dataset_name)
        if model == 'grpe_ffgt':
            self.collator = grpe_collator
        elif model == 'vanilla_ffgt':
            self.collator = vanilla_collator

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_dist = max_dist
        self.max_node = max_node
        self.max_freq = max_freq
        assert self.max_node <= self.dataset['max_node']
        
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

        # pre-compute
        if model == 'vanilla_ffgt':
            assert self.max_freq is not None
            if dataset_name in ['ZINC', 'voc-pixels'] or dataset_name.startswith('PATTERN') or dataset_name.startswith("CLUSTER"):
                for name in ['train_dataset', 'valid_dataset', 'test_dataset']:
                    pre_transform_in_memory(
                        self.dataset[name],
                        partial(
                            compute_LapPE,
                            max_freqs=max_freq
                        )
                    )
            elif dataset_name not in ["lsc-v2"]:
                pre_transform_in_memory(
                    self.dataset['dataset'],
                    partial(
                        compute_LapPE,
                        max_freqs=max_freq
                    )
                )

        self.setup(subset=subset)

    def setup(self, subset=False):
        if self.dataset_name in ["ZINC", "voc-pixels"] or self.dataset_name.startswith('PATTERN') or self.dataset_name.startswith('CLUSTER'):
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["valid_dataset"]
            self.dataset_test = self.dataset["test_dataset"]
        
        elif self.dataset_name == "lsc-v2":
            split_idx = self.dataset["dataset"].get_idx_split()
            rng = default_rng(seed=42)
            train_idx = rng.permutation(split_idx['train'].numpy())
            train_idx = torch.from_numpy(train_idx)
            valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
            if subset:
                subset_ratio = 0.1
                subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
                subvalid_idx = valid_idx[:50000]
                subtest_idx = split_idx['valid']

                self.dataset_train = self.dataset["dataset"][subtrain_idx]
                self.dataset_val = self.dataset["dataset"][subvalid_idx]
                self.dataset_test = self.dataset["dataset"][subtest_idx]
            else:
                self.dataset_train = self.dataset["dataset"][train_idx]
                self.dataset_val = self.dataset["dataset"][valid_idx]
                self.dataset_test = self.dataset["dataset"][split_idx['valid']]
            if self.model in ['vanilla_ffgt']:
                for data in [self.dataset_train, self.dataset_val, self.dataset_test]:
                    pre_transform_in_memory(
                        data,
                        partial(
                            compute_LapPE,
                            max_freqs=self.max_freq
                        )
                    )
        elif self.dataset_name in ["pep-func", "pep-struc"]:
            split_idx = self.dataset["dataset"].get_idx_split()
            self.dataset_train = self.dataset["dataset"][split_idx["train"]]
            self.dataset_val = self.dataset["dataset"][split_idx["val"]]
            self.dataset_test = self.dataset["dataset"][split_idx["test"]]
        else:
            split_idx = self.dataset["dataset"].get_idx_split()
            self.dataset_train = self.dataset["dataset"][split_idx["train"]]
            self.dataset_val = self.dataset["dataset"][split_idx["valid"]]
            self.dataset_test = self.dataset["dataset"][split_idx["test"]]

    def train_dataloader(self):

        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(
                self.collator,
                max_node=self.max_node,
                max_dist=self.max_dist,
                num_virtural_tokens=self.num_vitural,
                one_hot_edge=self.one_hot_edge
            )
        )
        print("len(train_dataloader)", len(loader))
        return loader

    def val_dataloader(self):

        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(
                self.collator,
                max_node=self.max_node,
                max_dist=self.max_dist,
                num_virtural_tokens=self.num_vitural,
                one_hot_edge=self.one_hot_edge
            )
        )
        print("len(val_dataloader)", len(loader))
        return loader

    def test_dataloader(self):

        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(
                self.collator,
                max_node=self.max_node,
                max_dist=self.max_dist,
                num_virtural_tokens=self.num_vitural,
                one_hot_edge=self.one_hot_edge
            )
        )
        print("len(test_dataloader)", len(loader))
        return loader
