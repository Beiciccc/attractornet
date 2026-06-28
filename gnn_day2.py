"""Day-2 make-or-break: does a GNN that READS THE BOOLEAN FUNCTIONS (truth tables)
beat the STRONG hand-crafted-feature GBDT (signed Thomas circuits + canalizing) at
predicting attractor properties -- especially in the logic-underdetermined regime
where the interaction graph provably cannot decide the label?

Self-contained, pure PyTorch (no PyG): Boolean nets are tiny (n<=16) so we use a
DENSE sign-aware message-passing GNN over a padded [B,N,N] signed adjacency.
Runs on CPU or Apple MPS. The GNN node features INCLUDE the padded truth table, so
the logic information is fully available to it -- if it still cannot beat the feature
GBDT, the within-fixed-wiring logic->attractor map is intrinsically hard (NO-GO);
if it does, the wedge is real and learnable (GO).

Tasks: has-cyclic (binary) | count (log regression). Eval on the standard ensemble
AND the same-graph/different-logic within-graph split (the provable-headroom regime).
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from attractor_gonogo import (gen_bn, nested_canalizing_tt, sync_attractors,
                              features, edge_sign)

KMAX = 5
TTLEN = 1 << KMAX
NMAX = 16
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SIGNED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]      # topology + Thomas signed circuits
LOGIC = SIGNED + [10, 11, 12, 13]            # + canalizing / bias (strong feature baseline)


def node_sens(tt, k):
    if k == 0:
        return 0.0
    flips = 0
    for x in range(1 << k):
        for b in range(k):
            if tt[x] != tt[x ^ (1 << b)]:
                flips += 1
    return flips / ((1 << k) * k)


def canal_depth(tt, k):
    d = 0
    for b in range(k):
        v0 = {tt[x] for x in range(len(tt)) if not ((x >> b) & 1)}
        v1 = {tt[x] for x in range(len(tt)) if ((x >> b) & 1)}
        if len(v0) == 1 or len(v1) == 1:
            d += 1
    return d


def to_graph(inputs, tt, n):
    """Dense padded representation: node features H [NMAX,F], signed adjacency A [NMAX,NMAX], mask."""
    A = np.zeros((NMAX, NMAX), np.float32)       # A[i,j] = sign of edge j->i (input j of node i)
    H = np.zeros((NMAX, 4 + TTLEN), np.float32)
    mask = np.zeros(NMAX, np.float32)
    for i in range(n):
        k = len(inputs[i])
        mask[i] = 1.0
        H[i, 0] = k / KMAX
        H[i, 1] = sum(tt[i]) / len(tt[i])                 # bias
        H[i, 2] = canal_depth(tt[i], k) / max(k, 1)
        H[i, 3] = node_sens(tt[i], k)
        H[i, 4:4 + len(tt[i])] = np.array(tt[i], np.float32)   # truth table (padded)
        for b, j in enumerate(inputs[i]):
            A[i, j] = edge_sign(inputs, tt, i, b)
    return H, A, mask


class SignGNN(nn.Module):
    def __init__(self, fin, h=64, layers=3, out=1):
        super().__init__()
        self.inp = nn.Linear(fin, h)
        self.ws = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wp = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wn = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, H, A, mask):
        Ap = torch.relu(A); An = torch.relu(-A)           # [B,N,N]
        x = torch.relu(self.inp(H))
        for ws, wp, wn in zip(self.ws, self.wp, self.wn):
            mp = torch.bmm(Ap, x); mn = torch.bmm(An, x)
            x = torch.relu(ws(x) + wp(mp) + wn(mn))
            x = x * mask.unsqueeze(-1)
        d = mask.sum(1, keepdim=True).clamp(min=1)
        mean = (x * mask.unsqueeze(-1)).sum(1) / d
        mx = (x + (mask.unsqueeze(-1) - 1) * 1e9).max(1).values
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


def build(n_list, K_list, modes, per_cell, seed):
    rng = random.Random(seed)
    Hs, As, Ms, X, ycyc, ycnt = [], [], [], [], [], []
    for n in n_list:
        for K in K_list:
            for mode in modes:
                for _ in range(per_cell):
                    inp, tt = gen_bn(n, K, mode, rng)
                    atts = sync_attractors(inp, tt, n)
                    H, A, m = to_graph(inp, tt, n)
                    Hs.append(H); As.append(A); Ms.append(m)
                    X.append(features(inp, tt, n))
                    ycyc.append(int(any(len(a) > 1 for a in atts)))
                    ycnt.append(len(atts))
    return (np.stack(Hs), np.stack(As), np.stack(Ms), np.array(X, float),
            np.array(ycyc), np.array(ycnt))


def train_gnn(Hs, As, Ms, y, tr, te, task, epochs=60, bs=128):
    Ht = torch.tensor(Hs, device=DEV); At = torch.tensor(As, device=DEV); Mt = torch.tensor(Ms, device=DEV)
    yt = torch.tensor(y.astype(np.float32), device=DEV)
    net = SignGNN(Hs.shape[-1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss() if task == "cyc" else nn.MSELoss()
    target = yt if task == "cyc" else torch.log1p(yt)
    tr_t = torch.tensor(tr, device=DEV)
    for ep in range(epochs):
        net.train(); perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = net(Ht[b], At[b], Mt[b])
            loss = lossf(out, target[b])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = net(Ht[torch.tensor(te, device=DEV)], At[torch.tensor(te, device=DEV)], Mt[torch.tensor(te, device=DEV)])
        pr = pr.float().cpu().numpy()
    if task == "cyc":
        return roc_auc_score(y[te], pr) if len(np.unique(y[te])) > 1 else float("nan")
    return r2_score(np.log1p(y[te]), pr)


def gbdt(X, y, tr, te, cols, task):
    if task == "cyc":
        if len(np.unique(y[tr])) < 2:
            return float("nan")
        c = HistGradientBoostingClassifier(max_iter=300).fit(X[tr][:, cols], y[tr])
        p = c.predict_proba(X[te][:, cols])[:, 1]
        return roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else float("nan")
    r = HistGradientBoostingRegressor(max_iter=300).fit(X[tr][:, cols], np.log1p(y[tr]))
    return r2_score(np.log1p(y[te]), r.predict(X[te][:, cols]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mode", choices=["ensemble", "withingraph"], default="ensemble")
    a = ap.parse_args()
    print(f"[gnn-day2] device={DEV} mode={a.mode}", flush=True)

    if a.mode == "ensemble":
        n_list, K_list, modes = [10, 12, 14, 16], [1, 2, 3, 4, 5], ["random", "canalizing"]
        Hs, As, Ms, X, ycyc, ycnt = build(n_list, K_list, modes, a.per_cell, 0)
        rs = np.random.RandomState(0); idx = rs.permutation(len(X))
        tr, te = idx[:int(.7*len(idx))], idx[int(.7*len(idx)):]
    else:
        # same-graph/different-logic: build fixed-wiring families
        rng = random.Random(7)
        n_list, K_list, GPC, V = [12, 14], [2, 3, 4], 80, 8
        Hs, As, Ms, X, ycyc, ycnt, gid, oig = [], [], [], [], [], [], [], []
        g = 0
        for n in n_list:
            for K in K_list:
                for _ in range(GPC):
                    inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                    for v in range(V):
                        mode = "random" if v % 2 == 0 else "canalizing"
                        tt = [nested_canalizing_tt(len(ins), rng) if mode == "canalizing"
                              else [1 if rng.random() < rng.uniform(.3, .7) else 0 for _ in range(1 << len(ins))]
                              for ins in inputs]
                        atts = sync_attractors(inputs, tt, n)
                        H, A, m = to_graph(inputs, tt, n)
                        Hs.append(H); As.append(A); Ms.append(m); X.append(features(inputs, tt, n))
                        ycyc.append(int(any(len(x) > 1 for x in atts))); ycnt.append(len(atts))
                        gid.append(g); oig.append(v)
                    g += 1
        Hs, As, Ms = np.stack(Hs), np.stack(As), np.stack(Ms)
        X, ycyc, ycnt, oig = np.array(X, float), np.array(ycyc), np.array(ycnt), np.array(oig)
        tr = np.where(oig % 2 == 0)[0]; te = np.where(oig % 2 == 1)[0]

    print(f"[gnn-day2] {len(X)} nets; train {len(tr)} test {len(te)}", flush=True)
    print(f"  {'task':<10}{'GNN(reads logic)':>18}{'GBDT signed-graph':>20}{'GBDT logic-aware':>18}", flush=True)
    for task, y in [("cyc", ycyc), ("cnt", ycnt)]:
        g_gnn = train_gnn(Hs, As, Ms, y, tr, te, task, epochs=a.epochs)
        g_sg = gbdt(X, y, tr, te, SIGNED, task)
        g_lo = gbdt(X, y, tr, te, LOGIC, task)
        metric = "AUROC" if task == "cyc" else "R2"
        print(f"  {task+'('+metric+')':<10}{g_gnn:>18.3f}{g_sg:>20.3f}{g_lo:>18.3f}", flush=True)
        print(f"      GNN minus strong feature baseline (logic-aware): {g_gnn - g_lo:+.3f}", flush=True)


if __name__ == "__main__":
    main()
