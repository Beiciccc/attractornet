"""Day-3a prerequisite: a SPARSE, size-agnostic, ARITY-INVARIANT sign-aware GNN.

Two changes from gnn_day2b so the model can TRANSFER from synthetic (n<=16) to real
BBM nets (n up to ~100, in-degree up to ~15):
  1. node features are ARITY-INVARIANT function descriptors (bias, canalizing depth,
     avg/max sensitivity, activating/inhibiting/non-monotone input fractions, degrees)
     -- defined for ANY in-degree, NO raw fixed-width truth table.
  2. message passing is sparse edge-list + scatter (no dense [N,N]) -> any graph size.

Prerequisite question: with arity-invariant features (transfer-ready), does the GNN
still BEAT the size/arity-agnostic summary GBDT on async reachability (the synthetic
win was GNN 0.962 with raw truth tables)? If yes, the transfer to real nets is
meaningful; if the raw truth table was load-bearing, the transfer story weakens.
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from attractor_gonogo import sync_attractors, features, edge_sign
from decisive_gate import make_logic, async_reachable, deconf_split, lgo_split

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def arity_inv_node_feats(inputs, tt, n, outdeg):
    """Per-node arity-invariant descriptors (fixed dim, any in-degree)."""
    F = np.zeros((n, 10), np.float32)
    for i in range(n):
        k = len(inputs[i]); table = tt[i]
        F[i, 0] = k
        F[i, 1] = k / 8.0
        F[i, 2] = sum(table) / len(table)                  # bias
        # canalizing depth + sensitivity + sign distribution
        depth = 0; act = inh = non = 0
        # sensitivity: exact if small else sampled
        if (1 << k) <= 4096:
            sflip = 0
            for x in range(1 << k):
                for b in range(k):
                    if table[x] != table[x ^ (1 << b)]:
                        sflip += 1
            sens = sflip / max((1 << k) * k, 1)
        else:
            rng = random.Random(i); sflip = tot = 0
            for _ in range(4096):
                x = rng.randrange(1 << k); b = rng.randrange(k)
                sflip += int(table[x] != table[x ^ (1 << b)]); tot += 1
            sens = sflip / max(tot, 1)
        maxs = 0.0
        for b in range(k):
            v0 = {table[x] for x in range(len(table)) if not ((x >> b) & 1)}
            v1 = {table[x] for x in range(len(table)) if ((x >> b) & 1)}
            if len(v0) == 1 or len(v1) == 1:
                depth += 1
            s = edge_sign(inputs, tt, i, b)
            if s > 0: act += 1
            elif s < 0: inh += 1
            else: non += 1
        F[i, 3] = depth
        F[i, 4] = depth / max(k, 1)
        F[i, 5] = sens
        F[i, 6] = act / max(k, 1)
        F[i, 7] = inh / max(k, 1)
        F[i, 8] = non / max(k, 1)
        F[i, 9] = outdeg[i]
    return F


def to_sparse(inputs, tt, n):
    outdeg = [0] * n
    for i in range(n):
        for j in inputs[i]:
            outdeg[j] += 1
    X = arity_inv_node_feats(inputs, tt, n, outdeg)
    src, dst, sgn = [], [], []
    for i in range(n):
        for b, j in enumerate(inputs[i]):
            src.append(j); dst.append(i); sgn.append(edge_sign(inputs, tt, i, b))
    return X, np.array(src), np.array(dst), np.array(sgn, np.float32)


class SparseSignGNN(nn.Module):
    def __init__(self, fin, h=64, layers=3):
        super().__init__()
        self.inp = nn.Linear(fin, h)
        self.ws = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wp = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wn = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, src, dst, sgn, batch, nG):
        x = torch.relu(self.inp(X))
        pos = torch.relu(sgn).unsqueeze(-1); neg = torch.relu(-sgn).unsqueeze(-1)
        deg = torch.zeros(x.size(0), 1, device=x.device).index_add_(
            0, dst, torch.ones(dst.size(0), 1, device=x.device)).clamp(min=1)
        for ws, wp, wn in zip(self.ws, self.wp, self.wn):
            mp = torch.zeros_like(x).index_add_(0, dst, x[src] * pos) / deg
            mn = torch.zeros_like(x).index_add_(0, dst, x[src] * neg) / deg
            x = torch.relu(ws(x) + wp(mp) + wn(mn))
        mean = torch.zeros(nG, x.size(1), device=x.device).index_add_(0, batch, x)
        cnt = torch.zeros(nG, 1, device=x.device).index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device)).clamp(min=1)
        mean = mean / cnt
        mx = torch.full((nG, x.size(1)), -1e9, device=x.device).scatter_reduce(
            0, batch.unsqueeze(-1).expand(-1, x.size(1)), x, reduce="amax", include_self=True)
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


def collate(graphs, idxs, dev):
    Xs, srcs, dsts, sgns, batch = [], [], [], [], []
    off = 0
    for bi, gi in enumerate(idxs):
        X, src, dst, sgn = graphs[gi]
        Xs.append(X); srcs.append(src + off); dsts.append(dst + off); sgns.append(sgn)
        batch.append(np.full(X.shape[0], bi)); off += X.shape[0]
    t = lambda a, dt: torch.tensor(np.concatenate(a), dtype=dt, device=dev)
    return (t(Xs, torch.float32), t(srcs, torch.long), t(dsts, torch.long),
            t(sgns, torch.float32), t(batch, torch.long), len(idxs))


def train_eval(graphs, y, tr, te, epochs=100, bs=256):
    net = SparseSignGNN(graphs[0][0].shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y.astype(np.float32), device=DEV)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            idxs = perm[i:i + bs]
            X, src, dst, sgn, batch, nG = collate(graphs, idxs, DEV)
            opt.zero_grad()
            out = net(X, src, dst, sgn, batch, nG)
            loss = lossf(out, yt[torch.tensor(idxs, device=DEV)])
            loss.backward(); opt.step()
    net.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(te), bs):
            idxs = te[i:i + bs]
            X, src, dst, sgn, batch, nG = collate(graphs, idxs, DEV)
            preds.append(net(X, src, dst, sgn, batch, nG).float().cpu().numpy())
    p = np.concatenate(preds)
    return roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else float("nan")


def build(task, n_list, K_list, gpc, v_each, seed=7):
    rng = random.Random(seed)
    graphs, Xs, y, gid, logic = [], [], [], [], []
    g = 0
    for n in n_list:
        s0, target = 0, (1 << n) - 1
        for K in K_list:
            for _ in range(gpc):
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        tt = make_logic(inputs, mode, rng)
                        lab = (int(any(len(a) > 1 for a in sync_attractors(inputs, tt, n))) if task == "cyc"
                               else async_reachable(inputs, tt, n, s0, target))
                        graphs.append(to_sparse(inputs, tt, n))
                        Xs.append(features(inputs, tt, n)); y.append(lab)
                        gid.append(g); logic.append(mode)
                g += 1
    return graphs, np.array(Xs, np.float32), np.array(y), np.array(gid), np.array(logic, object)


def gbdt(X, y, tr, te):
    if len(np.unique(y[tr])) < 2:
        return float("nan")
    c = HistGradientBoostingClassifier(max_iter=400).fit(X[tr], y[tr])
    p = c.predict_proba(X[te])[:, 1]
    return roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=100); a = ap.parse_args()
    print(f"[gnn-sparse arity-invariant] device={DEV} epochs={a.epochs}", flush=True)
    for task, (n_list, K_list) in {"cyc": ([12, 14], [2, 3, 4]), "reach": ([10, 12], [2, 3, 4])}.items():
        graphs, Xs, y, gid, logic = build(task, n_list, K_list, 80, 4)
        tr, te = deconf_split(gid, logic); trg, teg = lgo_split(gid)
        g_wg = train_eval(graphs, y, tr, te, epochs=a.epochs)
        g_lgo = train_eval(graphs, y, trg, teg, epochs=a.epochs)
        s_wg = gbdt(Xs, y, tr, te)
        print(f"\n== {task} (base {y.mean():.2f}) ==", flush=True)
        print(f"  within-graph:    GNN(arity-inv) {g_wg:.3f}  vs summary-GBDT {s_wg:.3f}  (Δ {g_wg-s_wg:+.3f})", flush=True)
        print(f"  leave-graph-out: GNN(arity-inv) {g_lgo:.3f}", flush=True)


if __name__ == "__main__":
    main()
