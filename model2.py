import torch
import torch.nn as nn
import math

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

# ---------- Flow-Matching head in trajectory space ----------
class FMSeq(nn.Module):
    """ f_theta(t, x_t, cond) -> R^{2H}, where H = horizon """
    def __init__(self, horizon=100, cond_dim=128, hidden=256):
        super().__init__()
        self.horizon = horizon
        act_dim = 2 * horizon  # full trajectory flattened
        self.net = nn.Sequential(
            nn.Linear(1 + act_dim + cond_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, t, x_t, cond):  # t:[B,1], x_t:[B,2H], cond:[B,C]
        return self.net(torch.cat([t, x_t, cond], dim=-1))

    def step(self, x_t, t_start, t_end, cond):
        """Midpoint-in-time (Heun) update"""
        dt = (t_end - t_start)
        t_mid = t_start.view(1,1).expand(x_t.size(0), 1) + 0.5*dt
        return x_t + dt * self.forward(t_mid, x_t, cond)

# ---------- Full model (uses FMSeq) ----------
class PolicySeqFlow(nn.Module):
    def __init__(self, ctx_dim=128, horizon=100, hidden=256):
        super().__init__()
        self.horizon = horizon
        self.pn  = PointNet1D(in_ch=8, emb_dims=128, out_ch=128)
        self.ctx = ContextNet(ego_dim=10, pn_dim=128, ctx_dim=ctx_dim)
        self.fm  = FMSeq(horizon=horizon, cond_dim=ctx_dim, hidden=hidden)

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


