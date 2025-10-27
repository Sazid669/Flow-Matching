import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module.
    Transforms a batch of time scalars to a batch of high-dimensional vectors.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): A tensor of time values, shape (batch_size,).
        Returns:
            torch.Tensor: The time embedding, shape (batch_size, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        # Create the frequency term
        # The original paper used 10000, but other values can work too.
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Calculate the arguments for sin and cos
        # t.unsqueeze(1) has shape (batch_size, 1)
        # emb.unsqueeze(0) has shape (1, half_dim)
        # The product has shape (batch_size, half_dim)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        # Concatenate sin and cos embeddings
        embedding = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # If dim is odd, pad with a zero
        if self.dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding
    

# ---------- PointNet encoder ----------
class PointNet1D(nn.Module):
    def __init__(self, in_ch=8, emb_dims=128, out_ch=128, momentum=0.1, drop_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 1, bias=False); self.bn1=nn.BatchNorm1d(64, momentum=momentum)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False);    self.bn2=nn.BatchNorm1d(64, momentum=momentum)
        self.conv3 = nn.Conv1d(64, 64, 1, bias=False);    self.bn3=nn.BatchNorm1d(64, momentum=momentum)
        self.conv4 = nn.Conv1d(64,128, 1, bias=False);    self.bn4=nn.BatchNorm1d(128, momentum=momentum)
        self.conv5 = nn.Conv1d(128,emb_dims,1,bias=False);self.bn5=nn.BatchNorm1d(emb_dims, momentum=momentum)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1  = nn.Linear(emb_dims, 256, bias=False); self.bn6=nn.BatchNorm1d(256, momentum=momentum)
        self.dp1  = nn.Dropout(drop_p); self.fc2=nn.Linear(256, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pts):  # pts: [B, Nobs=8, C=8]
        x = pts.transpose(1,2)                  # [B,C,Nobs]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x).squeeze(-1)            # [B, emb_dims]
        x = self.relu(self.bn6(self.fc1(x)))
        x = self.dp1(x)
        return self.fc2(x)                      # [B, out_ch=128]

# ---------- Context MLP ----------
class ContextSelfAttentionFromPN(nn.Module):
    """Self-attn over tokens: [CLS], EGO(10d), PN(128d). Returns ctx:[B, ctx_dim]."""
    def __init__(self, ego_dim=10, pn_dim=128,
                 d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                 ctx_dim=128):
        super().__init__()
        # token projections
        self.proj_ego = nn.Linear(ego_dim, d_model)
        self.proj_pn  = nn.Linear(pn_dim,  d_model)

        # learnable CLS + type embeddings (CLS=0, EGO=1, PN=2)
        self.cls_tok  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_tok, std=0.02) # one token broadcasted in batch with small random values
        self.type_emb = nn.Embedding(3, d_model)

        # transformer encoder (self-attention only)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)

        # CLS → ctx
        self.to_ctx = nn.Linear(d_model, ctx_dim)

    def forward(self, ego, pn_emb):
        """
        ego:    [B, 10]       (already includes lane in your pipeline)
        pn_emb: [B, 128]      (PointNet output)
        -> ctx: [B, ctx_dim]
        """
        B = ego.size(0)
        # build tokens
        cls = self.cls_tok.expand(B, 1, -1)             # [B,1,C]
        ego_tok = self.proj_ego(ego).unsqueeze(1)       # [B,1,C]
        pn_tok  = self.proj_pn(pn_emb).unsqueeze(1)     # [B,1,C]
        tokens  = torch.cat([cls, ego_tok, pn_tok], dim=1)  # [B,3,C]

        # add type embeddings
        type_ids = torch.tensor([0,1,2], device=ego.device).repeat(B,1)  # [B,3]
        tokens   = tokens + self.type_emb(type_ids)

        # self-attention over [CLS, EGO, PN]
        enc = self.encoder(tokens)                      # [B,3,C]
        ctx = enc.mean(dim=1)                           # Mean pooling 
        return self.to_ctx(ctx)                         # [B, ctx_dim]


# -----------Residual block-----------
class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU()
        )
    def forward(self, x):
        return x + self.block(x)


# ---------- Flow-Matching head in trajectory space ----------
class FMSeq(nn.Module):
    def __init__(self, horizon=100, cond_dim=128,
                 hidden=1024, te_dim=64):
        super().__init__()
        self.horizon = horizon
        act_dim = 2 * horizon
        in_dim  = te_dim + act_dim + cond_dim

        self.te   = SinusoidalTimeEmbedding(te_dim)
        self.fc_in  = nn.Linear(in_dim, hidden)
        self.act    = nn.ELU()
        self.fc_h1  = nn.Linear(hidden, hidden)
        self.fc_h2  = nn.Linear(hidden, hidden)
        self.fc_h3  = nn.Linear(hidden, hidden)

        # residual block using hidden dim
        self.res = ResidualBlock(hidden)

        self.fc_out = nn.Linear(hidden, act_dim)

    def forward(self, t, x_t, cond):
        if t.dim()==2 and t.size(-1)==1: t = t.squeeze(-1)
        te = self.te(t)
        h  = torch.cat([te, x_t, cond], dim=-1)

        h  = self.act(self.fc_in(h))
        h  = self.act(self.fc_h1(h))
        h  = self.act(self.fc_h2(h))
        h  = self.res(h)             
        h  = self.act(self.fc_h3(h))
        return self.fc_out(h)

    def step(self, x_t, t_start, t_end, cond):
        """Midpoint-in-time (Heun) update"""
        dt = (t_end - t_start)
        t_mid = t_start.view(1,1).expand(x_t.size(0), 1) + 0.5*dt
        return x_t + dt * self.forward(t_mid, x_t, cond)
    
    # ---------- Full model (uses FMSeq) ----------
class PolicySeqFlow(nn.Module):
    def __init__(self, ctx_dim=128, horizon=100, hidden=1024, te_dim=64,
                 ctx_d_model=128, ctx_heads=2, ctx_layers=1, ctx_dropout=0.1,
                 ego_dim=10, pn_dim=128):
        super().__init__()
        self.horizon = horizon
        self.pn  = PointNet1D(in_ch=8, emb_dims=128, out_ch=pn_dim)   # unchanged
        self.ctx_sa = ContextSelfAttentionFromPN(
            ego_dim=ego_dim, pn_dim=pn_dim,
            d_model=ctx_d_model, n_heads=ctx_heads, n_layers=ctx_layers,
            dropout=ctx_dropout, ctx_dim=ctx_dim
        )
        self.fm  = FMSeq(horizon=horizon, cond_dim=ctx_dim, hidden=hidden, te_dim=te_dim)

    def context(self, ego, obs):
        pn_emb = self.pn(obs)                  # [B, pn_dim]
        return self.ctx_sa(ego, pn_emb)        # [B, ctx_dim]

    def forward(self, ego, obs, t, x_t):
        ctx = self.context(ego, obs)
        return self.fm(t, x_t, ctx)
