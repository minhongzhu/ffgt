import logging
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional
from torch_geometric.datasets import Planetoid
import torch
import pdb
import numpy as np
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class PatternDataset(InMemoryDataset):
    r"""Several SBM-PATTERN Datasets generate with varied intra-community edge probability


    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"PATTERN_pq_010_001_005"`,
            :obj:`"PATTERN_pq_012_001_005"`, :obj:`"PATTERN_pq_014_001_005"`, 
            :obj:`"PATTERN_pq_016_001_005"`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #nodes
              - #degrees
              - #diameters
              - #features
              - #classes
            * - PATTERN_pq_010_001_005
              - 12,000
              - ~125
              - 6.13
              - 6.15
              - 3
              - 2
            * - PATTERN_pq_012_001_005
              - 12,000
              - 127
              - 5.72
              - 6.38
              - 3
              - 2
            * - PATTERN_pq_014_001_005
              - 12,000
              - 129
              - 5.34
              - 6.60
              - 3
              - 2
            * - PATTERN_pq_016_001_005
              - 12,000
              - 130
              - 4.85
              - 7.00
              - 3
              - 2
        
    """

    root_url = 'https://github.com/minhongzhu/sbm-pattern-new'
    urls = {
        'PATTERN_pq_010_001_005': f'{root_url}/pq_010_001_005/PATTERN_v2.pt',
        'PATTERN_pq_012_001_005': f'{root_url}/pq_012_001_005/PATTERN_v2.pt',
        'PATTERN_pq_014_001_005': f'{root_url}/pq_014_001_005/PATTERN_v2.pt',
        'PATTERN_pq_016_001_005': f'{root_url}/pq_016_001_005/PATTERN_v2.pt'
    }

    def __init__(self, root: str, name: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name.startswith('PATTERN')

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'PATTERN_v2.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        inputs = torch.load(self.raw_paths[0])
        for i in range(len(inputs)):
            data_list = [Data(**data_dict) for data_dict in inputs[i]]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[i])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
    

def GenerateShortestPath(data, directed=False):
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    weight = np.ones_like(row)

    graph = csr_matrix((weight, (row, col)), shape=(len(data.x), len(data.x)))
    # unreachable nodes will be -9999
    dist_matrix, _ = shortest_path(
        csgraph=graph, directed=directed, return_predecessors=True
    )

    return torch.from_numpy(dist_matrix)
    

if __name__ == '__main__':
    
    # a quick start with loading Pattern and check statistics
    
    dataset = PatternDataset('./dataset/SBM-PATTERN',"PATTERN_pq_010_001_005",split='train')
    max_dist = 0
    max_num = -1
    dis = 0
    total_num = []
    e_num = []
    n_num = []
    for i in range(10000):
        g = dataset.get(i)
        dist_mat = GenerateShortestPath(g)
        dist = torch.max(dist_mat).item()
        edge_num = len(g.edge_index[0])
        node_num = g.x.shape[0]
        e_num.append(edge_num)
        n_num.append(node_num)
        if dist == float('inf'):
            dis += 1
        elif dist > max_dist:
            total_num.append(dist)
            max_dist = dist
        else:
            total_num.append(dist)

    print(max_dist)
    print(dis)
    print('average diameter:', sum(total_num)/len(total_num))
    ave = sum(e_num)/len(e_num)
    avn = sum(n_num)/len(n_num)
    avd = ave / avn
    print('average edges:', ave)
    print('average nodes:', avn)
    print('average degree:', avd)
