import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import pdb

'''
    Utils for padding
    You have to make sure that graphs in a patch to have the same input shape
    attention: use "0" for padding token; make sure to distinguish it from other tokens
'''

# padding for edge_type
def pad_edge_attr_unsqueeze(x, padlen, one_hot_edge=True):
    xlen = x.size(0)
    if xlen < padlen:
        if one_hot_edge:
            new_x = - torch.ones([padlen, padlen], dtype=x.dtype)
            new_x[:xlen, :xlen] = x
            x = new_x
        else:
            new_x = - torch.ones([padlen, padlen, x.size(-1)], dtype=x.dtype)
            new_x[:xlen, :xlen, :] = x
            x = new_x
    return x.unsqueeze(0)

# padding for rel_pos
def pad_distance_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = - torch.ones([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

# padding for focal masks
def pad_focal_mask_bool(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = True # avoid nan in softmax
        x = new_x
    return x.unsqueeze(0)


# extra elements for Transformer
class extra_batch:
    def __init__(self, edge_attr, distance, focal_mask=None):
        super(extra_batch, self).__init__()
        
        self.edge_attr = edge_attr
        self.distance = distance
        self.focal_mask = focal_mask

    # put all these tensors to GPU together
    def to(self, device):
        self.edge_attr = self.edge_attr.to(device)
        self.distance = self.distance.to(device)
        if self.focal_mask is not None:
            self.focal_mask = self.focal_mask.to(device)
        return self

    def __len__(self):
        return self.distance.size(0)


# collate_fn in Dataloader
# do padding to ensure the same input length
def grpe_collator(items, max_node=128, max_dist=2, num_virtural_tokens=1, one_hot_edge=True):

    # num_virtual_tokens: 0 or 1; 1 for graph level tasks and 0 otherwise

    # MyDataset is an iterable object, you have to rewire and then collect graph items
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]

    # generate batch objects for gnn
    exclude_keys = ['distance', 'edge_attr', 'idx']
    batch = Batch.from_data_list(items, exclude_keys=exclude_keys)
    _, masks = to_dense_batch(batch.x, batch.batch)

    focal_masks = [
        torch.ones(
            [item.x.size(0) + num_virtural_tokens, item.x.size(0) + num_virtural_tokens],
            dtype=torch.bool
        )
        for item in items
    ]

    extra_attr = [
        (   
            item.edge_attr,
            item.distance
        )
        for item in items
    ]

    # group different items and process them seperately
    (
        edge_attrs,             # shape [n_nodes, n_nodes, c_edges]
        distances
    ) = zip(*extra_attr)

    if max_dist > 0:
        for idx, _ in enumerate(focal_masks):
            focal_masks[idx][num_virtural_tokens:, num_virtural_tokens:][
                distances[idx] > max_dist
            ] = False

    # Maximum number of nodes in the batch (not fixed, changed among batches).
    max_node_num = masks.shape[1]

    # edge attributes
    edge_attr = torch.cat([pad_edge_attr_unsqueeze(i, max_node_num, one_hot_edge) for i in edge_attrs])

    # distances
    distance = torch.cat([pad_distance_unsqueeze(i, max_node_num) for i in distances])

    # focal_masks
    if max_dist > 0:
        focal_mask = torch.cat([pad_focal_mask_bool(i, max_node_num + num_virtural_tokens) for i in focal_masks])
    else:
        focal_mask = None
        
    return batch, extra_batch(edge_attr, distance, focal_mask)
