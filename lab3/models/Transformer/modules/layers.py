import math

import torch
import torch.nn as nn


# TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d = dim // num_heads  # d_k, d_v are the same for efficiency
        self.num_heads = num_heads

        self.wqkv = nn.Linear(in_features=dim, out_features=(self.d * 3) * num_heads)
        self.wo = nn.Linear(in_features=dim, out_features=dim)
        self.drop = nn.Dropout(p=attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hint: input x tensor shape is (batch_size, num_image_tokens, dim),
        because the bidirectional transformer first will embed each token to dim dimension,
        and then pass to n_layers of encoders consist of Multi-Head Attention and MLP.
        # of head set 16
        Total d_k , d_v set to 768
        d_k , d_v for one head will be 768//16.
        """
        batch_size, num_image_tokens, dim = x.shape

        # last_dim: q_1, q_2, ..., q_h, k_1, k_2, ..., k_h, v_1, v_2, ..., v_h
        x = self.wqkv(x).view(batch_size, num_image_tokens, 3, self.num_heads, self.d)

        # (batch_size, num_heads, num_image_tokens, d)
        q, k, v = x.permute(2, 0, 3, 1, 4)
        o = (q @ k.mT / math.sqrt(self.d)).softmax(dim=-1)
        o = self.drop(o) @ v

        # (batch_size, num_image_tokens, num_heads, d)
        o = o.transpose(1, 2)

        # last_dim: o_h1, o_h2, ..., o_hh
        o = o.reshape(batch_size, num_image_tokens, dim)

        return self.wo(o)


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim), nn.Dropout(p=0.1))

    def forward(self, input):
        return super().forward(input)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim), nn.GELU(), nn.LayerNorm(dim, eps=1e-12)
        )

    def forward(self, input):
        return super().forward(input)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)

        x = x + attn
        x = self.LayerNorm1(x)

        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
