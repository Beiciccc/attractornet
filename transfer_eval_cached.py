"""Fast, repeatable synthetic->real transfer eval. Trains the synthetic GNN + per-node
GBDT, loads the cached real-net labels (real_cache.pkl from label_real_nets.py), and
reports pooled + per-net AUROC. Decoupled from the slow labeling.
"""
import argparse
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from day3_realtransfer import build_synth, train_gnn, gnn_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpc", type=int, default=40)
    ap.add_argument("--cache", default="real_cache.pkl")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    with open(a.cache, "rb") as f:
        real = pickle.load(f)
    Yr = np.concatenate([r["y"] for r in real])
    print(f"[eval] real nets={len(real)} nodes={len(Yr)} base={Yr.mean():.2f} "
          f"both-class nets={sum(1 for r in real if len(np.unique(r['y']))>1)}", flush=True)

    # synthetic data once; multi-seed GNN for a CI on the transfer gap
    graphs, Xnode, ynode = build_synth([12, 14], [2, 3, 4], a.gpc, 4)
    gb = HistGradientBoostingClassifier(max_iter=400).fit(Xnode, ynode)
    Xr = np.vstack([r["X"] for r in real])
    gb_auc = roc_auc_score(Yr, gb.predict_proba(Xr)[:, 1])

    gnn_pooled, gnn_pernet_med = [], []
    for s in range(a.seeds):
        np.random.seed(s)
        net = train_gnn(graphs, a.epochs)
        preds = [gnn_predict(net, (r["X"], r["src"], r["dst"], r["sgn"], r["y"])) for r in real]
        P = np.concatenate(preds)
        gnn_pooled.append(roc_auc_score(Yr, P))
        pernet = [roc_auc_score(r["y"], p) for r, p in zip(real, preds) if len(np.unique(r["y"])) > 1]
        gnn_pernet_med.append(float(np.median(pernet)))
        print(f"  seed {s}: GNN pooled {gnn_pooled[-1]:.3f} | per-net median {gnn_pernet_med[-1]:.3f}", flush=True)

    gp = np.array(gnn_pooled)
    print(f"\n================ SYNTHETIC->REAL TRANSFER ({len(real)} real GRNs, {len(Yr)} nodes) ================", flush=True)
    print(f"  GNN (size-agnostic, logic-aware)   pooled AUROC {gp.mean():.3f} ± {gp.std():.3f}  (seeds={a.seeds})", flush=True)
    print(f"  per-node-feature GBDT (size-agnostic) AUROC {gb_auc:.3f}", flush=True)
    print(f"  GNN - GBDT = {gp.mean()-gb_auc:+.3f}  | per-net GNN median {np.mean(gnn_pernet_med):.3f}", flush=True)
    print("  flat-TT GBDT: inapplicable at these sizes (variable n/arity).", flush=True)


if __name__ == "__main__":
    main()
