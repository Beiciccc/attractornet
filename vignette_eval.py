"""Real-GRN application vignette: pick a named BBM network, predict (zero-shot) which
genes are activatable from quiescence, validate against the exact symbolic oracle.
Reports candidates' size / base rate / per-net AUROC, then dumps the per-gene table
for the chosen network."""
import numpy as np
import torch
import biodivine_aeon as ba
from sklearn.metrics import roc_auc_score
from ablation_signaware import FlexGNN, train as train_gnn
from day3_pernode_syn import build as build_pernode
from amortization import cheap_feats
from day3_realtransfer import gnn_predict

CANDIDATES = ['10', '20', '40', '5', '13', '50', '8', '23', '24', '33']


def exact_labels(bn):
    g = ba.AsynchronousGraph(bn)
    names = bn.variable_names()
    init = g.mk_subspace({nm: False for nm in names})
    reached = ba.Reachability.reach_fwd(g, init).vertices()
    return np.array([int(not reached.intersect(g.mk_subspace_vertices({nm: True})).is_empty())
                     for nm in names], np.float32)


def main():
    graphs, *_ = build_pernode([12, 14], [2, 3, 4], 40, 4)
    np.random.seed(0); torch.manual_seed(0)
    net = train_gnn(graphs, np.arange(len(graphs)), "sign", 60)
    print("[vignette] GNN trained.\n", flush=True)

    rows = []
    for i in CANDIDATES:
        try:
            m = ba.BiodivineBooleanModels.fetch_model(i)
            bn = m.to_bn_inputs_false()
            n = bn.variable_count()
            if n > 60:
                continue
            y = exact_labels(bn)
            X, src, dst, sgn = cheap_feats(bn)
            p = gnn_predict(net, (X, src, dst, sgn, np.zeros(n, np.float32)))
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            rows.append((i, m.name, n, float(y.mean()), auc, bn, y, p))
            print(f"  id={i:>3} {m.name:<34} n={n:>3} base={y.mean():.2f} AUROC={auc:.3f}", flush=True)
        except Exception as e:
            print(f"  id={i} err {repr(e)[:50]}", flush=True)

    # pick: non-degenerate base in [0.25,0.75], highest AUROC, recognizable name
    cand = [r for r in rows if 0.2 <= r[3] <= 0.8 and r[4] == r[4]]
    cand.sort(key=lambda r: -r[4])
    if not cand:
        print("\nno non-degenerate candidate"); return
    i, name, n, base, auc, bn, y, p = cand[0]
    names = bn.variable_names()
    print(f"\n================ VIGNETTE: {name} (BBM id {i}, n={n}) ================", flush=True)
    print(f"  base rate {base:.2f} | per-net AUROC {auc:.3f}", flush=True)
    order = np.argsort(-p)
    print(f"\n  {'gene':<16}{'GNN p(activatable)':>20}{'exact label':>14}", flush=True)
    print("  --- most confidently activatable ---", flush=True)
    for j in order[:8]:
        print(f"  {names[j][:16]:<16}{p[j]:>20.3f}{int(y[j]):>14}", flush=True)
    print("  --- most confidently NOT activatable ---", flush=True)
    for j in order[-6:]:
        print(f"  {names[j][:16]:<16}{p[j]:>20.3f}{int(y[j]):>14}", flush=True)
    # agreement
    pred = (p > 0.5).astype(int)
    acc = (pred == y).mean()
    print(f"\n  thresholded (0.5) accuracy {acc:.2f}; {int(y.sum())}/{n} genes activatable (exact).", flush=True)
    try:
        print(f"  publication: {m.url_publication}", flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
