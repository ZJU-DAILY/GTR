from dgl.nn.pytorch import GATConv
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_dim, out_dim, num_heads=num_heads)

    def forward(self, g, inputs):
        # 输入是节点的特征
        h = self.gat_conv(g, inputs)
        h = F.elu(h)
        return h

class GAT2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, d_model):
        super(GAT2, self).__init__()
        self.d_model = d_model
        self.layer1 = GATLayer(in_dim, hidden_dim, num_heads)
        # 注意这里乘以 num_heads，以适应多头注意力的输出维度
        self.layer2 = GATLayer(hidden_dim * num_heads, out_dim, 1)  # 假设第二层是单头

    def forward(self, g, features, x):
        h = self.layer1(g, features)
        # 对于多头输出，通常需要合并（如果是最后一层，可能会平均）
        h = h.view(h.size(0), -1)  # 重塑以合并头
        h = self.layer2(g, h)
        h = h.squeeze()
        batch_size, seq_len = x.shape
        # print(torch.count_nonzero(node_fea_emb))
        node_fea_emb = h.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        return out_node_fea_emb




