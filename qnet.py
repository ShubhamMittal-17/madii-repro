"""Q-network for the MADII reproduction.

MADII scores every (sensor, next-hop) pair: their Fig. 3 outputs a B x N x (N+1) table
of Q values from a transformer over the N sensors plus the sink. We reproduce that
shape. The published model uses an Informer encoder-decoder (ProbSparse attention,
d_model 1024, d_ff 4096). This file has both:

  arch="transformer"  -> plain multi-head attention  = their MADTI ablation
  arch="informer"     -> ProbSparse attention (top-u queries) = MADII proper

Q(i, j) is a scaled dot product between node i's query embedding and candidate j's key
embedding, which keeps the output table over all N+1 candidates without an N^2 head.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 6      # x, y, residual energy, distance to sink, alive flag, is-sink flag


class ProbSparseSelfAttention(nn.Module):
    """Informer's ProbSparse attention (Zhou et al., AAAI 2021), factor c.

    Measures how far each query's attention is from uniform (the paper's sparsity
    measurement M), keeps the top u = c*ln(L) queries, computes full attention only for
    those, and fills the rest with the mean of V. O(L log L) instead of O(L^2).
    """

    def __init__(self, d_model, n_head, factor=5, dropout=0.1):
        super().__init__()
        self.h, self.dk, self.factor = n_head, d_model // n_head, factor
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.q, self.k, self.v = nn.ModuleList([self.q, self.k, self.v])
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dk).transpose(1, 2)

        u = min(L, max(1, int(self.factor * math.ceil(math.log(L)))))
        n_sample = min(L, max(1, int(self.factor * math.ceil(math.log(L)))))
        idx = torch.randint(0, L, (n_sample,), device=x.device)
        scores_sample = torch.einsum("bhld,bhsd->bhls", q, k[:, :, idx, :]) / math.sqrt(self.dk)
        # sparsity measurement M(q, K) = max score - mean score
        M = scores_sample.max(-1).values - scores_sample.mean(-1)
        top = M.topk(u, dim=-1).indices                       # (B, h, u)

        q_red = torch.gather(q, 2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk))
        scores = torch.einsum("bhud,bhld->bhul", q_red, k) / math.sqrt(self.dk)
        ctx_red = torch.einsum("bhul,bhld->bhud", scores.softmax(-1), v)

        ctx = v.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()   # lazy queries
        ctx.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk), ctx_red)
        ctx = ctx.transpose(1, 2).reshape(B, L, D)
        return self.drop(self.out(ctx))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, arch, factor, dropout):
        super().__init__()
        if arch == "informer":
            self.attn = ProbSparseSelfAttention(d_model, n_head, factor, dropout)
        else:
            mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            self.attn = lambda x, _m=mha: _m(x, x, x, need_weights=False)[0]
            self.mha = mha
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x)))
        return self.n2(x + self.drop(self.ff(x)))


class QNet(nn.Module):
    def __init__(self, d_model=128, n_head=4, d_ff=256, n_layers=2,
                 arch="transformer", factor=5, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(N_FEAT, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, arch, factor, dropout)
                                     for _ in range(n_layers)])
        self.q_head, self.k_head = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.arch = arch

    def forward(self, feats):
        """feats: (B, N+1, N_FEAT) -> Q table (B, N, N+1)."""
        h = self.embed(feats)
        for lyr in self.layers:
            h = lyr(h)
        q, k = self.q_head(h[:, :-1, :]), self.k_head(h)
        return torch.einsum("bid,bjd->bij", q, k) / self.scale
