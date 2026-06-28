"""Day-3 (per-node task, synthetic re-confirmation before real-net transfer).

Task = PER-NODE ACTIVATION REACHABILITY: from the all-OFF state, can node v ever
become 1 under asynchronous dynamics? (node-classification). This is the scalable,
non-degenerate, biologically-meaningful task that transfers to real BBM nets via
aeon's symbolic reach_fwd(all-false) ∩ {v=1}. It is PROPAGATION-structured, so the
GNN's message passing should beat a per-node feature baseline that lacks context.

Re-confirm on synthetic (exact async BFS oracle) that the sparse arity-invariant GNN
with a PER-NODE head beats a GBDT on per-node arity-invariant features. If yes, the
transfer experiment to real nets is well-posed.
"""
import argparse
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from attractor_gonogo import node_next
from decisive_gate import make_logic
from gnn_sparse import to_sparse, DEV


def pernode_activation(inputs, tt, n):
    """Exact: from all-zeros, async-reachable set; label[v]=1 if some reachable state has bit v set.
    OR-accumulate reachable states (O(states), memory = the visited set only)."""
    s0 = 0
    seen = {s0}; dq = deque([s0]); reached_or = s0
    while dq:
        s = dq.popleft()
        for i in range(n):
            if node_next(s, i, inputs, tt) != ((s >> i) & 1):
                ns = s ^ (1 << i)
                if ns not in seen:
                    seen.add(ns); dq.append(ns); reached_or |= ns
    return [(reached_or >> v) & 1 for v in range(n)]


class PerNodeGNN(nn.Module):
    def __init__(self, fin, h=64, layers=3):
        super().__init__()
        self.inp = nn.Linear(fin, h)
        self.ws = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wp = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wn = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, src, dst, sgn):
        x = torch.relu(self.inp(X))
        pos = torch.relu(sgn).unsqueeze(-1); neg = torch.relu(-sgn).unsqueeze(-1)
        deg = torch.zeros(x.size(0), 1, device=x.device).index_add_(
            0, dst, torch.ones(dst.size(0), 1, device=x.device)).clamp(min=1)
        for ws, wp, wn in zip(self.ws, self.wp, self.wn):
            mp = torch.zeros_like(x).index_add_(0, dst, x[src] * pos) / deg
            mn = torch.zeros_like(x).index_add_(0, dst, x[src] * neg) / deg
            x = torch.relu(ws(x) + wp(mp) + wn(mn))
        return self.head(x).squeeze(-1)        # per-node logit


def collate(graphs, idxs, dev):
    Xs, srcs, dsts, sgns, ynode = [], [], [], [], []
    off = 0
    for gi in idxs:
        X, src, dst, sgn, y = graphs[gi]
        Xs.append(X); srcs.append(src + off); dsts.append(dst + off); sgns.append(sgn)
        ynode.append(y); off += X.shape[0]
    cat = lambda a, dt: torch.tensor(np.concatenate(a), dtype=dt, device=dev)
    return (cat(Xs, torch.float32), cat(srcs, torch.long), cat(dsts, torch.long),
            cat(sgns, torch.float32), cat(ynode, torch.float32))


def train_eval(graphs, tr, te, epochs=60, bs=128):
    net = PerNodeGNN(graphs[0][0].shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            idxs = perm[i:i + bs]
            X, src, dst, sgn, y = collate(graphs, idxs, DEV)
            opt.zero_grad(); out = net(X, src, dst, sgn)
            lossf(out, y).backward(); opt.step()
    net.eval(); P, Y = [], []
    with torch.no_grad():
        for i in range(0, len(te), bs):
            X, src, dst, sgn, y = collate(graphs, te[i:i + bs], DEV)
            P.append(net(X, src, dst, sgn).float().cpu().numpy()); Y.append(y.cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return roc_auc_score(Y, P) if len(np.unique(Y)) > 1 else float("nan")


def build(n_list, K_list, gpc, v_each, seed=7):
    rng = random.Random(seed)
    graphs, gid, logic = [], [], []
    Xnode, ynode = [], []                        # for per-node GBDT baseline
    g = 0
    for n in n_list:
        for K in K_list:
            for _ in range(gpc):
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        tt = make_logic(inputs, mode, rng)
                        X, src, dst, sgn = to_sparse(inputs, tt, n)
                        y = np.array(pernode_activation(inputs, tt, n), np.float32)
                        graphs.append((X, src, dst, sgn, y))
                        Xnode.append(X); ynode.append(y)
                        gid.append(g); logic.append(mode)
                g += 1
    return graphs, gid, np.array(logic, object), np.vstack(Xnode), np.concatenate(ynode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpc", type=int, default=50)
    a = ap.parse_args()
    print(f"[day3 per-node synthetic] device={DEV} gpc={a.gpc}", flush=True)
    graphs, gid, logic, Xnode, ynode = build([12, 14], [2, 3, 4], a.gpc, 4)
    gid = np.array(gid)
    # de-confounded within-graph split at GRAPH level (balanced logic), map to node rows for GBDT
    rs = np.random.RandomState(0); trg_idx, teg_idx = [], []
    for gg in np.unique(gid):
        for mode in ("random", "canalizing"):
            ix = np.where((gid == gg) & (logic == mode))[0]
            ix = rs.permutation(ix); h = len(ix) // 2
            trg_idx += list(ix[:h]); teg_idx += list(ix[h:])
    trg_idx, teg_idx = np.array(trg_idx), np.array(teg_idx)
    # node-row spans per graph-instance
    sizes = [g[0].shape[0] for g in graphs]
    starts = np.cumsum([0] + sizes)
    rows = lambda inst: np.arange(starts[inst], starts[inst + 1])
    tr_rows = np.concatenate([rows(i) for i in trg_idx]); te_rows = np.concatenate([rows(i) for i in teg_idx])

    base = ynode.mean()
    gnn = train_eval(graphs, trg_idx, teg_idx, epochs=a.epochs)
    c = HistGradientBoostingClassifier(max_iter=400).fit(Xnode[tr_rows], ynode[tr_rows])
    gb = roc_auc_score(ynode[te_rows], c.predict_proba(Xnode[te_rows])[:, 1])
    print(f"\n== per-node activation reachability (node base rate {base:.2f}) ==", flush=True)
    print(f"  GNN(arity-inv, per-node) {gnn:.3f}  vs  per-node-feature GBDT {gb:.3f}  (Δ {gnn-gb:+.3f})", flush=True)
    print("  GO if GNN clearly beats the per-node feature baseline (propagation context matters).", flush=True)


if __name__ == "__main__":
    main()
