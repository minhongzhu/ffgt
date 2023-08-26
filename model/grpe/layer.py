import torch
import torch.nn as nn
import pdb


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, att_size, attention_dropout_rate, num_heads, focal_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.focal_heads = focal_heads

        self.att_size = att_size 
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
        focal_mask=None
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        global_size = self.num_heads - self.focal_heads

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, d_k, k_len]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.num_heads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        spatial_bias = query_hop + key_hop
        edge_bias = query_edge + key_edge

        attn = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bias
        attn = attn * self.scale

        if mask is not None:
            if focal_mask is not None:
                focal_mask = focal_mask.unsqueeze(1)
                attn1 = attn[:,:global_size,:,:].masked_fill(
                    ~mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf")
                )
                attn2 = attn[:,global_size:,:,:].masked_fill(~focal_mask, float("-inf"))
                attn = torch.cat([attn1, attn2], dim=1)
            else:
                attn = attn.masked_fill(
                    ~mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf")
                )

        attn = torch.softmax(attn, dim=3)
        attn = self.att_dropout(attn)

    
        value_hop_emb = value_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        value_edge_emb = value_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)

        value_hop_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_hop_types),
            device=value_hop_emb.device,
        )
        value_hop_att = torch.scatter_add(
            value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1), attn
        )
        value_edge_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_edge_types),
            device=value_hop_emb.device,
        )
        value_edge_att = torch.scatter_add(
            value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1), attn
        )

        x = (
            torch.matmul(attn, v)
            + torch.matmul(value_hop_att, value_hop_emb)
            + torch.matmul(value_edge_att, value_edge_emb)
        )
        

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, attn_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, focal_heads
    ):
        super(EncoderLayer, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_size)

        # attention layer
        self.attention = MultiHeadAttention(
            hidden_size,
            attn_size,
            attention_dropout_rate,
            num_heads,
            focal_heads
        )

        # attention dropout
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask,
        focal_mask
    ):
        # main Transformer
        y = self.attention_norm(x)
        y = self.attention(
            y,
            y,
            y,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            value_hop_emb,
            value_edge_emb,
            distance,
            edge_attr,
            mask=mask,
            focal_mask=focal_mask
        )
        y = self.attention_dropout(y)

        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
