from Model.embedding import *
import math
import torch.nn.functional as F
from Model.transformer import Block
from functools import partial


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        self.attn = 0

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        self.attn = attn
        return self.output_linear(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



class VITwithGAT(nn.Module):
    def __init__(self, args, vocab_size, adj, feature, alpha=1):
        super().__init__()
        self.ModelEmbedding = ModelEmbedding_GAT_Fusion(vocab_size=vocab_size, grid_size=args.grid_size, d_model=args.d_model, max_len=args.max_len, adj=adj, feature=feature)
        # Configure the norm_layer to be used in Blocks
        configured_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(
                dim=args.d_model,
                num_heads=args.n_heads,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=configured_norm_layer,  # Use the pre-configured norm layer
                proj_drop=0.2,  # 添加 dropout
                attn_drop=0.2,  # 添加路径 dropout
                drop_path=0.2
            )
            for _ in range(args.n_layers)
        ])
        print(f'model_embedding d_model = {args.d_model}, vit model n_layers = {args.n_layers}, n_heads = {args.n_heads}')

    def forward(self, x, min_list, weekday_list, day_list, grid_list, poi_list, task_list):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.ModelEmbedding(x, min_list, weekday_list, day_list, grid_list, poi_list, task_list)
        for layer in self.blocks:
            x, attn = layer(x)
        return x

class TrajBERT(nn.Module):
    def __init__(self, args, bert, vocab_size):
        super().__init__()
        self.model = bert
        self.predictTokenLinear = nn.Linear(args.d_model, vocab_size)
        self.predictTimeLinear = nn.Linear(args.d_model, 1)
        self.simplifyLinear = nn.Linear(args.d_model, 2)  # 2-classification
        self.ClassificationLinear = nn.Linear(args.d_model, args.num_class)
        self.pretrain_classificationLinear = nn.Linear(args.d_model, 2)
        self.abnormal_detect_Linear = nn.Linear(args.d_model, 2)
        self.imputationLinear = nn.Linear(args.d_model, vocab_size)
        self.destinationLinear = nn.Linear(args.d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, min_list, weekday_list, day_list, grid_list, poi_list, task_list, task):
        x = self.model(x, min_list, weekday_list, day_list, grid_list, poi_list, task_list)
        if task == 'pretrain_mlm':
            mlm_token = self.softmax(self.predictTokenLinear(x))
            return mlm_token
        elif task == 'trj_predict':
            token_predict = self.softmax(self.predictTokenLinear(x))
            return token_predict
        elif task == 'classification':
            result = self.ClassificationLinear(x[:, 0:1, :])
            return result
        elif task == 'time_estimate':
            result = self.predictTimeLinear(x[:, 0:1, :])
            return result
        elif task == 'similarity':
            return x
        elif task == 'simplify':
            result = self.simplifyLinear(x)
            return result
        elif task == 'imputation':
            result = self.softmax(self.imputationLinear(x))
            return result
        else:
            return x
