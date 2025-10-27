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
class ContextNet(nn.Module):
    def __init__(self, ego_dim=10, pn_dim=128, ctx_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ego_dim + pn_dim, 256), nn.ReLU(),
            nn.Linear(256, ctx_dim), nn.ReLU(),
        )
    def forward(self, ego, pn):                 
        return self.net(torch.cat([ego, pn], dim=-1))  # [B,ctx_dim]

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


# --- Simple cross-attention block ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, ctx_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.ctx_proj = nn.Linear(ctx_dim, q_dim)   # project context to q_dim

    def forward(self, x_t, ctx):
        """
        x_t : [B, H, q_dim]     (noisy vel+steer sequence)
        ctx : [B, ctx_dim]      (ego + PN context vector)
        """
        # expand context → sequence length 1
        ctx_seq = self.ctx_proj(ctx).unsqueeze(1)   # [B, 1, q_dim]
        out, _ = self.attn(query=x_t, key=ctx_seq, value=ctx_seq)
        return out + x_t   # residual skip


# ---------- Flow-Matching head in trajectory space ----------
class FMSeq(nn.Module):
    def __init__(self, horizon=100, cond_dim=128,
                 hidden=1024, te_dim=64,
                 ca_q=128, ca_heads=2, ca_dropout=0.1, ca_res_scale=0.5):
        super().__init__()
        self.horizon = horizon
        act_dim = 2 * horizon
        in_dim  = te_dim + act_dim + cond_dim

        # --- time embedding ---
        self.te = SinusoidalTimeEmbedding(te_dim)

        # --- original MLP trunk ---
        self.fc_in  = nn.Linear(in_dim, hidden)
        self.act    = nn.ELU()
        self.fc_h1  = nn.Linear(hidden, hidden)
        self.fc_h2  = nn.Linear(hidden, hidden)
        self.res    = ResidualBlock(hidden)
        self.fc_h3  = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, act_dim)

        # --- cross-attention branch: queries = [x_step(2) + te], K/V = ctx ---
        self.q_proj   = nn.Linear(2 + te_dim, ca_q)   # per-step query dim
        self.ctx_proj = nn.Linear(cond_dim, ca_q)     # project ctx to K/V dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ca_q, num_heads=ca_heads, dropout=ca_dropout, batch_first=True
        )
        # pre-norms for stability with residuals
        self.ln_q  = nn.LayerNorm(ca_q)
        self.ln_kv = nn.LayerNorm(ca_q)

        # fuse attended sequence back to MLP width
        self.ca_to_hidden = nn.Linear(horizon * ca_q, hidden)
        self.ca_res_scale = ca_res_scale  # scale for cross-attn residual

    def forward(self, t, x_t, cond):
        # t: [B,1] or [B], x_t: [B, 2H], cond: [B, cond_dim]
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        te = self.te(t)  # [B, te_dim]

        # ---- main MLP path ----
        h = torch.cat([te, x_t, cond], dim=-1)   # [B, te_dim + 2H + cond_dim]
        h = self.act(self.fc_in(h))
        h = self.act(self.fc_h1(h))
        h = self.act(self.fc_h2(h))
        h = self.res(h)                          

        # ---- cross-attention branch ----
        B, H = x_t.size(0), self.horizon
        x_steps = x_t.view(B, H, 2)                              # [B, H, 2]
        te_rep  = te.unsqueeze(1).expand(B, H, te.size(-1))      # [B, H, te_dim]
        q = self.q_proj(torch.cat([x_steps, te_rep], dim=-1))    # [B, H, ca_q]

        # K/V from context (length-1 sequence)
        kv = self.ctx_proj(cond).unsqueeze(1)                    # [B, 1, ca_q]

        # pre-norm + cross-attn
        qn  = self.ln_q(q)
        kvn = self.ln_kv(kv)
        ca_out, _ = self.cross_attn(query=qn, key=kvn, value=kvn)  # [B, H, ca_q]

        # fuse attended sequence back to the flat MLP stream
        h = h + self.ca_res_scale * self.ca_to_hidden(ca_out.reshape(B, -1))
        h = self.act(self.fc_h3(h))
        return self.fc_out(h)   # [B, 2H]

    def step(self, x_t, t_start, t_end, cond):
        """Midpoint-in-time (Heun) update"""
        dt = (t_end - t_start)
        t_mid = t_start.view(1,1).expand(x_t.size(0), 1) + 0.5*dt
        return x_t + dt * self.forward(t_mid, x_t, cond)
    
    # ---------- Full model (uses FMSeq) ----------
class PolicySeqFlow(nn.Module):
    def __init__(self, ctx_dim=128, horizon=100, hidden=256, te_dim=64):
        super().__init__()
        self.horizon = horizon
        self.pn  = PointNet1D(in_ch=8, emb_dims=128, out_ch=128)
        self.ctx = ContextNet(ego_dim=10, pn_dim=128, ctx_dim=ctx_dim)
        self.fm  = FMSeq(horizon=horizon, cond_dim=ctx_dim, hidden=hidden, te_dim=te_dim)

    def context(self, ego, obs):
        emb = self.pn(obs)         
        return self.ctx(ego, emb)

    # Optional convenience forward if I need single call
    def forward(self, ego, obs, t, x_t):
        """Returns v_theta(t, x_t, ctx). Shapes:
           ego:[B,10], obs:[B,8,8], t:[B,1], x_t:[B,2H] -> [B,2H]
        """
        ctx = self.context(ego, obs)
        return self.fm(t, x_t, ctx)