"""Stage B: load the real-net cache, train the per-node activation-reachability
predictor on SYNTHETIC nets, evaluate ZERO-SHOT transfer to the cached real GRNs.
Reports pooled AUROC (GNN vs size-agnostic per-node GBDT) over the class-varying
nets, per-net median, and coverage stats. Fast + repeatable (no re-fetch/re-label).
"""
import argparse
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from day3_realtransfer import build_synth, train_gnn, gnn_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="real_cache.pkl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpc", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    cache = pickle.load(open(a.cache, "rb"))
    both = [r for r in cache if r["both"] == 1]
    print(f"[eval] cache={len(cache)} nets, {sum(r['both'] for r in cache)} with both classes; "
          f"sizes n={sorted(set(r['n'] for r in cache))[:12]}...", flush=True)

    # real pooled arrays over class-varying nets
    Yr = np.concatenate([r["y"] for r in both])
    Xr = np.vstack([r["X"] for r in both])
    print(f"[eval] evaluating transfer on {len(both)} class-varying nets, {len(Yr)} nodes, base {Yr.mean():.2f}", flush=True)

    gnn_aucs, gb_aucs, gnn_pernet = [], [], []
    for s in range(a.seeds):
        np.random.seed(s)
        graphs, Xnode, ynode = build_synth([12, 14], [2, 3, 4], a.gpc, 4, seed=7 + s)
        net = train_gnn(graphs, a.epochs)
        gb = HistGradientBoostingClassifier(max_iter=400).fit(Xnode, ynode)
        # pooled GNN preds
        Pr = np.concatenate([gnn_predict(net, (r["X"], r["src"], r["dst"], r["sgn"], r["y"])) for r in both])
        gnn_aucs.append(roc_auc_score(Yr, Pr))
        gb_aucs.append(roc_auc_score(Yr, gb.predict_proba(Xr)[:, 1]))
        per = [roc_auc_score(r["y"], gnn_predict(net, (r["X"], r["src"], r["dst"], r["sgn"], r["y"]))) for r in both]
        gnn_pernet.append(np.median(per))
        print(f"  seed {s}: GNN {gnn_aucs[-1]:.3f} | GBDT {gb_aucs[-1]:.3f} | per-net-median {gnn_pernet[-1]:.3f}", flush=True)

    g, gb_ = np.array(gnn_aucs), np.array(gb_aucs)
    print(f"\n================ REAL-BBM TRANSFER ({len(both)} nets, {len(Yr)} nodes) ================", flush=True)
    print(f"  GNN  AUROC  {g.mean():.3f} ± {g.std():.3f}", flush=True)
    print(f"  GBDT AUROC  {gb_.mean():.3f} ± {gb_.std():.3f}", flush=True)
    print(f"  GNN - GBDT  {(g-gb_).mean():+.3f} ± {(g-gb_).std():.3f}   (per-net GNN median {np.mean(gnn_pernet):.3f})", flush=True)
    print(f"  flat-TT GBDT: inapplicable at these sizes (variable n / arity).", flush=True)


if __name__ == "__main__":
    main()
