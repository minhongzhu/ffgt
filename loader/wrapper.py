import random
import torch
import numpy as np
import pdb

import torch_geometric.datasets
from torch_geometric.datasets import GNNBenchmarkDataset
from ogb.graphproppred import PygGraphPropPredDataset
from lrgb.peptides_functional import PeptidesFunctionalDataset
from lrgb.peptides_structural import PeptidesStructuralDataset
from loader.pattern import PatternDataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

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

def OneHotEdgeAttr(data, max_range=4):
    x = data.edge_attr
    if len(x.shape) == 1:
        return data

    offset = torch.ones((1, x.shape[1]), dtype=torch.long)
    offset[:, 1:] = max_range
    offset = torch.cumprod(offset, dim=1)
    x = (x * offset).sum(dim=1)
    return x

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item, name=None):
    # item: Data(x, edge_index, edge_attr, y) ; the raw graph data

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)

    N = x.size(0)

    # node feature
    if name not in ['PATTERN', 'CLUSTER']:
        x = convert_to_single_emb(x)  # For ZINC: [n_nodes, 1]
    else:
        x = x.float()

    # distance
    distance = GenerateShortestPath(item).long()

    # edge feature 
    if name in ['zinc','PATTERN', 'CLUSTER']:
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        edge_attr_2d = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        edge_attr_2d[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )
        edge_attr_2d = edge_attr_2d.squeeze(-1)
    elif name in ['voc-pixels']:
        # edge_attr are float rather than integer
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        edge_attr_2d = -torch.ones([N, N, edge_attr.size(-1)], dtype=torch.float)
        edge_attr_2d[edge_index[0, :], edge_index[1, :]] = edge_attr
    else:
        '''
            ogb format dataset
        '''
        edge_attr = OneHotEdgeAttr(item)
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        edge_attr_2d = -torch.ones([N, N, edge_attr.size(-1)], dtype=torch.long)
        edge_attr_2d[edge_index[0, :], edge_index[1, :]] = (
            edge_attr
        )
        edge_attr_2d = edge_attr_2d.squeeze(-1)

    item.x = x                                      # shape [n_nodes,c_nodes] (c=1 for "ZINC")                                      
    item.edge_attr = edge_attr_2d                # shape [n_nodes, n_nodes, c_edges](c=1 for "ZINC")
    item.distance = distance                        # shape [n_nodes, n_nodes]; 

    return item


class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, name='zinc')
        else:
            return self.index_select(idx)


class MyGNNBenchmarkDataset(GNNBenchmarkDataset):
    def download(self):
        super(MyGNNBenchmarkDataset, self).download()

    def process(self):
        super(MyGNNBenchmarkDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, name=self.name)
        else:
            return self.index_select(idx)


class MyPatternDataset(PatternDataset):
    def download(self):
        super(MyPatternDataset, self).download()

    def process(self):
        super(MyPatternDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, name='PATTERN')
        else:
            return self.index_select(idx)


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPeptidesFunctionalDataset(PeptidesFunctionalDataset):
    def download(self):
        super(MyPeptidesFunctionalDataset, self).download()

    def process(self):
        super(MyPeptidesFunctionalDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, name='pep-func')
        else:
            return self.index_select(idx)


class MyPeptidesStructuralDataset(PeptidesStructuralDataset):
    def download(self):
        super(MyPeptidesStructuralDataset, self).download()

    def process(self):
        super(MyPeptidesStructuralDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, 'pep-struc')
        else:
            return self.index_select(idx)

        

        



