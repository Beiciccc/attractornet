"""A (sign-aware ablation) + B (multi-seed CIs) on the headline per-node activation-
reachability task. Three GNN variants isolate the source of the GNN's advantage:
  sign  : separate transforms for +/- edges (the proposed model)
  nosign: single aggregation over all neighbours, sign IGNORED
  mlp   : no message passing (per-node MLP) ~ a neural per-node-feature baseline
Reports in-distribution within-graph AUROC mean±std over seeds, plus GBDT, plus
ZERO-SHOT transfer to real BBM nets (real_cache.pkl) for sign vs nosign.
If sign >> nosign (in-dist AND transfer), the sign-aware propagation is load-bearing.
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from gnn_sparse import DEV
from day3_pernode_syn import build as build_pernode, collate


class FlexGNN(nn.Module):
    def __init__(self, fin, mode, h=64, layers=3):
        super().__init__()
        self.mode = mode
        self.inp = nn.Linear(fin, h)
        self.ws = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wp = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.wn = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, src, dst, sgn):
        x = torch.relu(self.inp(X))
        if self.mode != "mlp":
            deg = torch.zeros(x.size(0), 1, device=x.device).index_add_(
                0, dst, torch.ones(dst.size(0), 1, device=x.device)).clamp(min=1)
            pos = torch.relu(sgn).unsqueeze(-1); neg = torch.relu(-sgn).unsqueeze(-1)
        for li in range(len(self.ws)):
            if self.mode == "sign":
                mp = torch.zeros_like(x).index_add_(0, dst, x[src] * pos) / deg
                mn = torch.zeros_like(x).index_add_(0, dst, x[src] * neg) / deg
                x = torch.relu(self.ws[li](x) + self.wp[li](mp) + self.wn[li](mn))
            elif self.mode == "nosign":
                m = torch.zeros_like(x).index_add_(0, dst, x[src]) / deg
                x = torch.relu(self.ws[li](x) + self.wp[li](m))
            else:                                   # mlp
                x = torch.relu(self.ws[li](x))
        return self.head(x).squeeze(-1)


def train(graphs, idxs, mode, epochs, bs=128):
    net = FlexGNN(graphs[0][0].shape[1], mode).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idxs)
        for i in range(0, len(perm), bs):
            X, s, d, g, y = collate(graphs, perm[i:i + bs], DEV)
            opt.zero_grad(); lossf(net(X, s, d, g), y).backward(); opt.step()
    return net


def predict(net, graphs, idxs, bs=128):
    net.eval(); P, Y = [], []
    with torch.no_grad():
        for i in range(0, len(idxs), bs):
            X, s, d, g, y = collate(graphs, idxs[i:i + bs], DEV)
            P.append(net(X, s, d, g).float().cpu().numpy()); Y.append(y.cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def deconf_graph_split(gid, logic, seed):
    rs = np.random.RandomState(seed); tr, te = [], []
    for gg in np.unique(gid):
        for mode in ("random", "canalizing"):
            ix = np.where((gid == gg) & (logic == mode))[0]
            ix = rs.permutation(ix); h = len(ix) // 2
            tr += list(ix[:h]); te += list(ix[h:])
    return np.array(tr), np.array(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--gpc", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    print(f"[ablation A+B] device={DEV} seeds={a.seeds}", flush=True)

    # build once (graphs + per-node feats); labels are deterministic
    graphs, gid, logic, Xnode, ynode = build_pernode([12, 14], [2, 3, 4], a.gpc, 4)
    gid = np.array(gid)
    sizes = [g[0].shape[0] for g in graphs]; starts = np.cumsum([0] + sizes)
    rows = lambda S: np.concatenate([np.arange(starts[i], starts[i + 1]) for i in S])

    res = {m: [] for m in ("sign", "nosign", "mlp")}; gbdt = []
    for s in range(a.seeds):
        np.random.seed(s); torch.manual_seed(s)
        tr, te = deconf_graph_split(gid, logic, s)
        for m in ("sign", "nosign", "mlp"):
            net = train([graphs[i] for i in tr], np.arange(len(tr)), m, a.epochs)
            P, Y = predict(net, graphs, te)
            res[m].append(roc_auc_score(Y, P))
        c = HistGradientBoostingClassifier(max_iter=400).fit(Xnode[rows(tr)], ynode[rows(tr)])
        gbdt.append(roc_auc_score(ynode[rows(te)], c.predict_proba(Xnode[rows(te)])[:, 1]))
        print(f"  seed {s}: sign {res['sign'][-1]:.3f} nosign {res['nosign'][-1]:.3f} "
              f"mlp {res['mlp'][-1]:.3f} | GBDT {gbdt[-1]:.3f}", flush=True)

    print("\n=== A+B: in-distribution per-node activation reachability (within-graph, mean±std) ===", flush=True)
    for m in ("sign", "nosign", "mlp"):
        arr = np.array(res[m]); print(f"  GNN-{m:<7} {arr.mean():.3f} ± {arr.std():.3f}", flush=True)
    g = np.array(gbdt); print(f"  per-node GBDT  {g.mean():.3f} ± {g.std():.3f}", flush=True)
    sa, ns = np.array(res['sign']), np.array(res['nosign'])
    print(f"  >>> sign − nosign = {(sa-ns).mean():+.3f} ± {(sa-ns).std():.3f}  "
          f"(is sign-aware message passing load-bearing?)", flush=True)

    # transfer ablation on real nets
    try:
        cache = pickle.load(open("real_cache.pkl", "rb"))
        both = [r for r in cache if r["both"] == 1]
        Yr = np.concatenate([r["y"] for r in both])
        from day3_realtransfer import build_synth, gnn_predict
        tr_sign, tr_ns = [], []
        for s in range(a.seeds):
            np.random.seed(s); torch.manual_seed(s)
            gA, _, _ = build_synth([12, 14], [2, 3, 4], a.gpc, 4, seed=7 + s)
            for mode, store in (("sign", tr_sign), ("nosign", tr_ns)):
                net = train(gA, np.arange(len(gA)), mode, a.epochs)
                Pr = np.concatenate([gnn_predict_flex(net, r) for r in both])
                store.append(roc_auc_score(Yr, Pr))
        print(f"\n=== transfer to {len(both)} real nets (mean±std) ===", flush=True)
        print(f"  GNN-sign   {np.mean(tr_sign):.3f} ± {np.std(tr_sign):.3f}", flush=True)
        print(f"  GNN-nosign {np.mean(tr_ns):.3f} ± {np.std(tr_ns):.3f}", flush=True)
        print(f"  >>> sign − nosign on REAL = {(np.array(tr_sign)-np.array(tr_ns)).mean():+.3f}", flush=True)
    except FileNotFoundError:
        print("  (real_cache.pkl not found; skipping transfer ablation)", flush=True)


def gnn_predict_flex(net, r):
    cat = lambda arr, dt: torch.tensor(arr, dtype=dt, device=DEV)
    net.eval()
    with torch.no_grad():
        return net(cat(r["X"], torch.float32), cat(r["src"], torch.long),
                   cat(r["dst"], torch.long), cat(r["sgn"], torch.float32)).float().cpu().numpy()


if __name__ == "__main__":
    main()
