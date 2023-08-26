import torch
import pdb

# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))


class AtomEncoder(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        # from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        """
            x: [batch_size, num_nodes, feature_dim]
        """
        encoded_features = 0
        for i in range(x.shape[2]):
            encoded_features += self.atom_embedding_list[i](x[:, :, i])

        x = encoded_features
        return x


class BondEncoder(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """
    def __init__(self, emb_dim):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        """"""
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        edge_attr = bond_embedding
        return edge_attr