"""A (sign-aware ablation) + B (multi-seed CIs): does the GNN's advantage come from
SIGN-AWARE message passing (separate transforms for activating/inhibiting edges)?

For each seed: train a SIGN-AWARE GNN, a SIGN-AGNOSTIC GNN (single message, ignores
edge sign), and a per-node-feature GBDT on synthetic per-node activation reachability
(inputs-OFF task). Report in-distribution within-graph AUROC and REAL-GRN transfer
AUROC (from real_cache.pkl), mean ± std over seeds. This isolates the inductive-bias
claim and puts CIs on the headline comparisons.
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from gnn_sparse import DEV
from day3_pernode_syn import collate
from day3_realtransfer import build_synth


class UnifiedGNN(nn.Module):
    def __init__(self, fin, h=64, layers=3, sign_aware=True):
        super().__init__()
        self.sa = sign_aware
        self.inp = nn.Linear(fin, h)
        self.ws = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wp = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wn = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))  # unused if not sa
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, src, dst, sgn):
        x = torch.relu(self.inp(X))
        deg = torch.zeros(x.size(0), 1, device=x.device).index_add_(
            0, dst, torch.ones(dst.size(0), 1, device=x.device)).clamp(min=1)
        if self.sa:
            pos = torch.relu(sgn).unsqueeze(-1); neg = torch.relu(-sgn).unsqueeze(-1)
            for ws, wp, wn in zip(self.ws, self.wp, self.wn):
                mp = torch.zeros_like(x).index_add_(0, dst, x[src] * pos) / deg
                mn = torch.zeros_like(x).index_add_(0, dst, x[src] * neg) / deg
                x = torch.relu(ws(x) + wp(mp) + wn(mn))
        else:                                           # SIGN-AGNOSTIC: one message, sign ignored
            for ws, wp in zip(self.ws, self.wp):
                m = torch.zeros_like(x).index_add_(0, dst, x[src]) / deg
                x = torch.relu(ws(x) + wp(m))
        return self.head(x).squeeze(-1)


def train(graphs, idxs, sign_aware, epochs, bs=128):
    net = UnifiedGNN(graphs[0][0].shape[1], sign_aware=sign_aware).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = np.random.permutation(idxs)
        for i in range(0, len(perm), bs):
            X, s, d, g, y = collate(graphs, perm[i:i + bs], DEV)
            opt.zero_grad(); lossf(net(X, s, d, g), y).backward(); opt.step()
    return net


def predict(net, graph):
    X, src, dst, sgn, y = graph
    cat = lambda a, dt: torch.tensor(a, dtype=dt, device=DEV)
    with torch.no_grad():
        return net(cat(X, torch.float32), cat(src, torch.long), cat(dst, torch.long), cat(sgn, torch.float32)).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpc", type=int, default=40)
    a = ap.parse_args()
    cache = pickle.load(open("real_cache.pkl", "rb"))
    both = [r for r in cache if r["both"] == 1]
    Yr = np.concatenate([r["y"] for r in both]); Xr = np.vstack([r["X"] for r in both])
    res = {k: {"indist": [], "transfer": []} for k in ["sign-aware GNN", "sign-agnostic GNN", "GBDT"]}
    for s in range(a.seeds):
        np.random.seed(s)
        graphs, Xnode, ynode = build_synth([12, 14], [2, 3, 4], a.gpc, 4, seed=7 + s)
        idx = np.random.permutation(len(graphs)); tr, te = idx[:int(.7*len(idx))], idx[int(.7*len(idx))::]
        starts = np.cumsum([0] + [g[0].shape[0] for g in graphs])
        rows = lambda S: np.concatenate([np.arange(starts[i], starts[i + 1]) for i in S])
        Yte = np.concatenate([graphs[i][4] for i in te])
        for sa, name in [(True, "sign-aware GNN"), (False, "sign-agnostic GNN")]:
            net = train(graphs, tr, sa, a.epochs)
            P = np.concatenate([predict(net, graphs[i]) for i in te])
            res[name]["indist"].append(roc_auc_score(Yte, P))
            Pr = np.concatenate([predict(net, (r["X"], r["src"], r["dst"], r["sgn"], r["y"])) for r in both])
            res[name]["transfer"].append(roc_auc_score(Yr, Pr))
        gb = HistGradientBoostingClassifier(max_iter=400).fit(Xnode[rows(tr)], ynode[rows(tr)])
        res["GBDT"]["indist"].append(roc_auc_score(ynode[rows(te)], gb.predict_proba(Xnode[rows(te)])[:, 1]))
        res["GBDT"]["transfer"].append(roc_auc_score(Yr, gb.predict_proba(Xr)[:, 1]))
        print(f"  seed {s} done", flush=True)
    print(f"\n========= SIGN ABLATION + CIs ({a.seeds} seeds; real transfer on {len(both)} nets) =========", flush=True)
    print(f"  {'model':<20}{'in-dist AUROC':>18}{'real-transfer AUROC':>22}", flush=True)
    for name in ["sign-aware GNN", "sign-agnostic GNN", "GBDT"]:
        ind, tr = np.array(res[name]["indist"]), np.array(res[name]["transfer"])
        print(f"  {name:<20}{ind.mean():>10.3f} ± {ind.std():.3f}{tr.mean():>14.3f} ± {tr.std():.3f}", flush=True)
    sa, sg = np.array(res["sign-aware GNN"]["transfer"]), np.array(res["sign-agnostic GNN"]["transfer"])
    print(f"\n  sign-aware − sign-agnostic (transfer): {(sa-sg).mean():+.3f} ± {(sa-sg).std():.3f}", flush=True)
    print("  -> if sign-aware clearly beats sign-agnostic, the signed message passing is load-bearing.", flush=True)


if __name__ == "__main__":
    main()
