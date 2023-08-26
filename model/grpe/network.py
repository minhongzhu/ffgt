import torch
import torch.nn as nn
import pdb

from torch_geometric.utils import to_dense_batch

from model.grpe.layer import EncoderLayer
from encoder.mol_encoder import AtomEncoder

def load_grpe_ffgt_backbone(model, load, skip_task_branch=True, map_location="cpu"):
    new_dict = {}
    for k, v in torch.load(load, map_location=map_location).items():
        if k.startswith("module."):
            k = k[7:]
        if (
            k.endswith("linear.weight") or k.endswith("linear.bias") \
                or k.endswith("last_mlp.0.0.weight") or k.endswith("last_mlp.0.0.bias")   
        ) and skip_task_branch:
            print("Skip loading the weight of task branch")
        else:
            new_dict[k] = v
    print(f"Loaded following keys {list(new_dict.keys())}")
    model.load_state_dict(new_dict, strict=False)

class Grpe_FFGT(nn.Module):
    def __init__(
        self,
        num_task=1,
        num_layer=10,
        d_model=80,
        d_attn=10,
        nhead=8,
        khead=4,
        dim_feedforward=80,
        dropout=0.1,
        attention_dropout=0.1,
        max_hop=64,
        num_node_type=25,
        num_edge_type=25,
        use_independent_token=False,
        perturb_noise=0.0,
        num_last_mlp=0,
        task='graph'
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        self.max_hop = max_hop
        self.num_edge_type = num_edge_type
        self.use_independent_token = use_independent_token
        self.task = task
        if task == 'graph':
            self.task_token = nn.Embedding(1, d_model, padding_idx=-1)

        if num_node_type < 0:   # node features are not integer
            num_node_type = -num_node_type
            self.node_emb = nn.Linear(num_node_type, d_model)
        else:
            self.node_emb = nn.Embedding(num_node_type, d_model)
            # self.node_emb = AtomEncoder(d_model)
        
        if num_edge_type < 0:
            num_edge_type = - num_edge_type
            self.edge_emb = nn.Linear(num_edge_type, d_model)

        self.TASK_DISTANCE = max_hop + 1
        self.UNREACHABLE_DISTANCE = max_hop + 2

        # assume that edge features is of shape [n_edge, 1]
        if num_edge_type > 0:
            self.TASK_EDGE = num_edge_type + 1
            self.SELF_EDGE = num_edge_type + 2
            self.NO_EDGE = num_edge_type + 3

        # query_hop_emb: Query Structure Embedding
        # query_edge_emb: Query Edge Embedding
        # key_hop_emb: Key Structure Embedding
        # key_edge_emb: Key Edge Embedding
        # value_hop_emb: Value Structure Embedding
        # value_edge_emb: Value Edge Embedding

        if not self.use_independent_token:
            self.query_hop_emb = nn.Embedding(max_hop + 3, nhead * d_attn)
            self.key_hop_emb = nn.Embedding(max_hop + 3, nhead * d_attn)
            self.value_hop_emb = nn.Embedding(max_hop + 3, nhead * d_attn)
            self.query_edge_emb = nn.Embedding(num_edge_type + 4, nhead * d_attn)
            self.key_edge_emb = nn.Embedding(num_edge_type + 4, nhead * d_attn)
            self.value_edge_emb = nn.Embedding(num_edge_type + 4, nhead * d_attn)

        else:
            self.query_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 3, nhead * d_attn) for _ in range(num_layer)]
            )
            self.query_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 4, nhead * d_attn) for _ in range(num_layer)]
            )
            self.key_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 3, nhead * d_attn) for _ in range(num_layer)]
            )
            self.key_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 4, nhead * d_attn) for _ in range(num_layer)]
            )
            self.value_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 3, nhead * d_attn) for _ in range(num_layer)]
            )
            self.value_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 4, nhead * d_attn) for _ in range(num_layer)]
            )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=d_model,
                    attn_size=d_attn,
                    ffn_size=dim_feedforward,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout,
                    num_heads=nhead,
                    focal_heads=khead,
                )
                for _ in range(num_layer)
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)
        self.last_mlp = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
                for _ in range(num_last_mlp)
            ]
        )
        self.linear = nn.Linear(d_model, num_task)

    def encode_node(self, batch):
        if isinstance(self.node_emb, nn.Linear):
            batch.x = self.node_emb(batch.x)
            return batch
        else:
            batch.x = self.node_emb.weight[batch.x].sum(dim=1)
            # batch.x = self.node_emb(batch.x)
            return batch

    def forward(self, batch, extra_batch):
        
        batch = self.encode_node(batch)
        x, mask = to_dense_batch(batch.x, batch.batch)
        del batch
        focal_mask = extra_batch.focal_mask
        
        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        # Append Task Token
        if self.task == 'graph':
            # add virtural node mask
            mask_new = torch.ones([mask.shape[0], mask.shape[1] + 1], dtype=torch.bool)
            mask_new[:, 1:] = mask
            mask = mask_new.clone()
            mask = mask.to(x.device)
            del mask_new

            x_with_task = torch.zeros(
                (x.shape[0], x.shape[1] + 1, x.shape[2]), dtype=x.dtype, device=x.device
            )
            x_with_task[:, 1:] = x
            x_with_task[:, 0] = self.task_token.weight

            distance_with_task = torch.zeros(
                (
                    extra_batch.distance.shape[0],
                    extra_batch.distance.shape[1] + 1,
                    extra_batch.distance.shape[2] + 1,
                ),
                dtype=extra_batch.distance.dtype,
                device=extra_batch.distance.device,
            )
            distance_with_task[:, 1:, 1:] = extra_batch.distance.clamp(
                max=self.max_hop
            )  
            distance_with_task[:, 0, 1:] = self.TASK_DISTANCE
            distance_with_task[:, 1:, 0] = self.TASK_DISTANCE
            distance_with_task[distance_with_task < 0] = self.UNREACHABLE_DISTANCE

            edge_attr_with_task = torch.zeros(
                (
                    extra_batch.edge_attr.shape[0],
                    extra_batch.edge_attr.shape[1] + 1,
                    extra_batch.edge_attr.shape[2] + 1,
                ),
                dtype=extra_batch.edge_attr.dtype,
                device=extra_batch.edge_attr.device,
            )
            edge_attr_with_task[:, 1:, 1:] = extra_batch.edge_attr
            edge_attr_with_task[
                :, range(edge_attr_with_task.shape[1]), range(edge_attr_with_task.shape[2])
            ] = self.SELF_EDGE
            edge_attr_with_task[edge_attr_with_task == -1] = self.NO_EDGE
            edge_attr_with_task[distance_with_task != 1] = self.NO_EDGE
            edge_attr_with_task[distance_with_task == self.TASK_DISTANCE] = self.TASK_EDGE
        else:
            x_with_task = x

            distance_with_task = extra_batch.distance.clamp(
                max=self.max_hop
            )
            distance_with_task[distance_with_task < 0] = self.UNREACHABLE_DISTANCE

            edge_attr_with_task = extra_batch.edge_attr
            edge_attr_with_task[
                :, range(edge_attr_with_task.shape[1]), range(edge_attr_with_task.shape[2])
            ] = self.SELF_EDGE
            edge_attr_with_task[edge_attr_with_task == -1] = self.NO_EDGE
            edge_attr_with_task[distance_with_task != 1] = self.NO_EDGE

        del extra_batch
        
        for i, enc_layer in enumerate(self.layers):
            if self.use_independent_token:
                x_with_task = enc_layer(
                    x_with_task,
                    self.query_hop_emb[i].weight,
                    self.query_edge_emb[i].weight,
                    self.key_hop_emb[i].weight,
                    self.key_edge_emb[i].weight,
                    self.value_hop_emb[i].weight,
                    self.value_edge_emb[i].weight,
                    distance_with_task,
                    edge_attr_with_task,
                    mask=mask,
                    focal_mask=focal_mask
                )
            else:
                x_with_task = enc_layer(
                    x_with_task,
                    self.query_hop_emb.weight,
                    self.query_edge_emb.weight,
                    self.key_hop_emb.weight,
                    self.key_edge_emb.weight,
                    self.value_hop_emb.weight,
                    self.value_edge_emb.weight,
                    distance_with_task,
                    edge_attr_with_task,
                    mask=mask,
                    focal_mask=focal_mask
                )

        if self.task == 'graph':
            output = self.final_ln(x_with_task[:, 0])
            output = self.last_mlp(output)  
            output = self.linear(output)
        elif self.task == 'inductive-node':
            output = self.final_ln(x_with_task)[mask]
            output = self.last_mlp(output)
            output = self.linear(output)

        return output
    
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

