import torch.nn as nn
import torch
import math
import numpy as np
from Model.GAT import GAT2
import dgl


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size):
        super().__init__(vocab_size, embedding_size, padding_idx=0)


class GridEmbedding(nn.Embedding):
    def __init__(self, grid_size, embedding_size):
        super().__init__(grid_size, embedding_size, padding_idx=0)

class TaskEmbedding(nn.Embedding):
    def __init__(self, task_size, embedding_size):
        super().__init__(task_size, embedding_size, padding_idx=0)

class POIEmbedding(nn.Embedding):
    def __init__(self, poi_size, embedding_size):
        super().__init__(poi_size, embedding_size, padding_idx=0)


class dayTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.day_embedding = nn.Embedding(367, d_model, padding_idx=0)

    def forward(self, daytime, weekday, day):
        return self.weekday_embedding(weekday) + self.daytime_embedding(daytime) + self.day_embedding(day)


class ModelEmbedding_GAT2(nn.Module):
    def __init__(self, vocab_size, grid_size, max_len, d_model, adj, feature):
        super().__init__()
        self.token_embedding = GAT2(
            in_dim=4, hidden_dim=16, out_dim=768, num_heads=8, d_model=d_model
        )
        self.token_embedding_2 = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.time_embedding = dayTimeEmbedding(d_model)
        self.grid_embedding = GridEmbedding(grid_size + 10, d_model)
        self.poi_embedding = POIEmbedding(poi_size=5, max_len=max_len)
        self.adj = self.convert_adj_edge_index(adj)
        self.feature = feature

    def convert_adj_edge_index(self, adj):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge = adj.nonzero().t()
        g = dgl.graph((edge[0], edge[1]))
        g = dgl.add_self_loop(g)
        return g.to(device)

    def forward(self, trj_token, min_list, weekday_list, day_list, grid_list, poi_list):
        token = self.token_embedding(self.adj, self.feature, trj_token)
        time_emb = self.time_embedding(min_list, weekday_list, day_list)
        grid_emb = self.grid_embedding(grid_list)
        pe = self.position_embedding(trj_token)
        return pe + token + time_emb + grid_emb


class ModelEmbedding_GAT_Fusion(nn.Module):
    def __init__(self, vocab_size, grid_size, max_len, d_model, adj, feature):
        super().__init__()
        self.token_embedding = GAT2(
            in_dim=4, hidden_dim=16, out_dim=768, num_heads=8, d_model=d_model
        )
        self.token_embedding_2 = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.time_embedding = dayTimeEmbedding(d_model)
        self.grid_embedding = GridEmbedding(grid_size + 10, d_model)
        self.task_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.poi_embedding = POIEmbedding(5, d_model)
        self.adj = self.convert_adj_edge_index(adj)
        self.feature = feature
        self.spatial_temporal_fusion = MoEFusion(d_model, d_model) # task_dim, temporal_dim, spatial_dim, hidden_dim, output_dim

    def convert_adj_edge_index(self, adj):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge = adj.nonzero().t()
        g = dgl.graph((edge[0], edge[1]))
        g = dgl.add_self_loop(g)
        return g.to(device)

    def forward(self, trj_token, min_list, weekday_list, day_list, grid_list, poi_list, task_list):
        token = self.token_embedding(self.adj, self.feature, trj_token)
        time_emb = self.time_embedding(min_list, weekday_list, day_list)
        grid_emb = self.grid_embedding(grid_list)
        pe = self.position_embedding(trj_token)
        poi_emb = self.poi_embedding(poi_list)

        spatial_info = pe + token + grid_emb + poi_emb
        temporal_info = time_emb
        task_emb = self.task_embedding(task_list)
        fused_output = self.spatial_temporal_fusion(task_emb, temporal_info, spatial_info)
        return fused_output


class GatingNetwork(nn.Module):
    def __init__(self, task_dim, hidden_dim, num_experts=2, top_k=1):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(task_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.top_k = top_k
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, task_emb):
        """
        task_emb: [batch_size, task_dim]
        Returns:
            weights: [batch_size, num_experts]
        """
        x = self.relu(self.fc1(task_emb))  # [batch_size, hidden_dim]
        logits = self.fc2(x)            # [batch_size, num_experts]
        if self.top_k == 2:
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
            mask = torch.zeros_like(logits).scatter(1, top_k_indices, 1.0)
            masked_logits = logits * mask + (-1e9) * (1 - mask)
            weights = self.softmax(masked_logits)  # [batch_size, num_experts]
        elif self.top_k == 1:
            weights = self.softmax(logits)  # [batch_size, num_experts]
        else:
            raise ValueError("Unsupported top_k value")

        return weights

class TemporalFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=6)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, temporal_emb):
        temporal_emb = temporal_emb.transpose(0, 1)  # 调整维度以匹配nn.MultiheadAttention的输入
        attn_output, _ = self.attn(temporal_emb, temporal_emb, temporal_emb)
        attn_output = self.dropout(attn_output)
        output = self.norm(temporal_emb + attn_output)
        return output.transpose(0, 1)


class SpatialFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=6)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, spatial_emb):
        spatial_emb = spatial_emb.transpose(0, 1)
        attn_output, _ = self.attn(spatial_emb, spatial_emb, spatial_emb)
        attn_output = self.dropout(attn_output)
        output = self.norm(spatial_emb + attn_output)
        return output.transpose(0, 1)


class MoEFusion(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_experts=2, top_k=1):
        super(MoEFusion, self).__init__()
        self.gating = GatingNetwork(task_dim=hidden_dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
        self.temporal_expert = TemporalFusion(d_model=hidden_dim)
        self.spatial_expert = SpatialFusion(d_model=hidden_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, task_emb, temporal_emb, spatial_emb):
        """
        task_emb: [batch_size, task_dim]
        temporal_emb: [batch_size, seq_len, temporal_dim]
        spatial_emb: [batch_size, seq_len, spatial_dim]
        Returns:
            fused_output: [batch_size, seq_len, output_dim]
        """
        weights = self.gating(task_emb)  # [batch_size, 2]

        # 专家处理
        temporal_out = self.temporal_expert(temporal_emb)  # [batch_size, seq_len, temporal_dim]
        spatial_out = self.spatial_expert(spatial_emb)  # [batch_size, seq_len, spatial_dim]

        # 确保 temporal_dim 和 spatial_dim 一致，或者通过线性层调整
        if temporal_out.size(-1) != spatial_out.size(-1):
            raise ValueError("Temporal and Spatial expert outputs must have the same dimension")

        # 加权组合
        fused_output = weights[:, 0].unsqueeze(1).unsqueeze(2) * temporal_out + \
                       weights[:, 1].unsqueeze(1).unsqueeze(2) * spatial_out  # [batch_size, seq_len, d_model]

        # 残差连接和层归一化
        fused_output = self.layer_norm(fused_output + self.dropout(fused_output))

        return fused_output


