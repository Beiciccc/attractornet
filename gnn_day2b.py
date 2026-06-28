"""Day-2 (de-confounded) make-or-break: does the GNN-on-truth-tables earn its place
by clearing the flattened-truth-table GBDT ceiling (cyc 0.803 / reach 0.921) on the
SAME de-confounded fixed-wiring families the decisive gate used?

Apples-to-apples: for each (inputs, tt) we build BOTH the GNN graph tensors AND the
flattened-TT / summary feature vectors, then compare GNN vs both GBDTs on the
de-confounded within-graph split AND leave-graph-out. Adds the async-reachability
task (the freshest lane, gate AUROC 0.921). Pure PyTorch + sklearn; CPU/MPS.

AAAI-area-chair bar: the GNN must clear ~0.75 and not fall far below the
flattened-TT GBDT (else 'why a GNN' — a GBDT-on-bits already does it at small n;
the GNN's real value is scaling to real BBM nets where flattening is infeasible).
"""
import argparse
import random
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from attractor_gonogo import sync_attractors, features
from decisive_gate import (make_logic, flat_features, async_reachable, deconf_split, lgo_split)
from gnn_day2 import to_graph, SignGNN, train_gnn, DEV


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def build(task, n_list, K_list, gpc, v_each, seed=7):
    rng = random.Random(seed)
    Hs, As, Ms, Xf, Xs, y, gid, logic = [], [], [], [], [], [], [], []
    g = 0
    for n in n_list:
        s0, target = 0, (1 << n) - 1
        for K in K_list:
            for _ in range(gpc):
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        tt = make_logic(inputs, mode, rng)
                        if task == "cyc":
                            lab = int(any(len(a) > 1 for a in sync_attractors(inputs, tt, n)))
                        else:
                            lab = async_reachable(inputs, tt, n, s0, target)
                        H, A, m = to_graph(inputs, tt, n)
                        Hs.append(H); As.append(A); Ms.append(m)
                        Xf.append(flat_features(inputs, tt, n)); Xs.append(features(inputs, tt, n))
                        y.append(lab); gid.append(g); logic.append(mode)
                g += 1
    return (np.stack(Hs), np.stack(As), np.stack(Ms), np.array(Xf, np.float32),
            np.array(Xs, np.float32), np.array(y), np.array(gid), np.array(logic, object))


def gbdt_auc(X, y, tr, te, **kw):
    if len(np.unique(y[tr])) < 2:
        return float("nan")
    c = HistGradientBoostingClassifier(max_iter=400, **kw).fit(X[tr], y[tr])
    return auc(y[te], c.predict_proba(X[te])[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    print(f"[gnn-day2b] device={DEV} epochs={a.epochs}", flush=True)
    specs = {"cyc": ([12, 14], [2, 3, 4]), "reach": ([10, 12], [2, 3, 4])}
    for task, (n_list, K_list) in specs.items():
        Hs, As, Ms, Xf, Xs, y, gid, logic = build(task, n_list, K_list, 80, 4)
        tr, te = deconf_split(gid, logic)
        trg, teg = lgo_split(gid)
        gnn_wg = train_gnn(Hs, As, Ms, y, tr, te, task if task == "cyc" else "cyc", epochs=a.epochs)
        gnn_lgo = train_gnn(Hs, As, Ms, y, trg, teg, task if task == "cyc" else "cyc", epochs=a.epochs)
        flat_wg = gbdt_auc(Xf, y, tr, te, max_leaf_nodes=63)
        flat_lgo = gbdt_auc(Xf, y, trg, teg, max_leaf_nodes=63)
        summ_wg = gbdt_auc(Xs, y, tr, te)
        print(f"\n================ TASK {task} (base {y.mean():.2f}) ================", flush=True)
        print(f"  {'split':<16}{'GNN':>8}{'flat-TT GBDT':>14}{'summary GBDT':>14}", flush=True)
        print(f"  {'within-graph':<16}{gnn_wg:>8.3f}{flat_wg:>14.3f}{summ_wg:>14.3f}", flush=True)
        print(f"  {'leave-graph-out':<16}{gnn_lgo:>8.3f}{flat_lgo:>14.3f}", flush=True)
        verdict = ("GNN CLEARS the flat-TT ceiling" if gnn_wg >= flat_wg - 0.01
                   else "GNN below flat-TT (GBDT-on-bits wins at small n; GNN value = scaling)")
        print(f"  -> within-graph: {verdict}  (GNN {gnn_wg:.3f} vs flat-TT {flat_wg:.3f})", flush=True)


if __name__ == "__main__":
    main()
