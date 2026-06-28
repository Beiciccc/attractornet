"""DECISIVE KILL-OR-KEEP GATE (CPU, no GNN, no GPU) — built from the adversarial panel.

The panel (4 skeptics) converged: before building any GNN, settle whether ANY
learnable logic->attractor signal exists, and fix the confounds in control_logic.py.

This script runs, for has-cyclic (sync) AND async target-state reachability (the
fresh un-tested lane), on fixed-wiring / different-logic families:

  FIX 1 (confound): de-confounded within-graph split — train and test each get a
        50/50 random+canalizing logic mix per graph (control_logic.py wrongly put
        all-random in train, all-canalizing in test => a hidden distribution shift).
  FIX 2 (ceiling proxy): a GBDT on the FULL FLATTENED TRUTH TABLES (every node's
        truth-table bits + signed wiring) — this contains LITERALLY ALL the logic
        bits a GNN could read. If it can't beat ~0.63, a GNN provably can't either.
  CONTROL A: canalizing-ONLY families (realistic biology) — does the label-flip
        rate across logic collapse (statistics skeptic's attack)?
  CONTROL B: observed flip rate vs the i.i.d. base-rate null (is 85% just label
        entropy, not structured headroom?).
  CONTROL C: within-graph label-shuffle null (should be ~0.5).
  CONTROL D: leave-graph-out split (no wiring overlap) — does signal survive?

VERDICT per task on the de-confounded flattened-truth-table within-graph AUROC:
  KILL    <= 0.63  (all logic bits present, still stuck => intrinsically unlearnable)
  GO      >= 0.75  (extractable logic signal a strong feature set captures => GNN worth it)
  PIVOT   in between
"""
import random
from collections import deque
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error
from attractor_gonogo import (nested_canalizing_tt, sync_attractors, features,
                              edge_sign, node_next)

NMAX, KMAX, TTLEN = 14, 5, 1 << 5


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def rand_tt(k, rng):
    p = rng.uniform(0.3, 0.7)
    return [1 if rng.random() < p else 0 for _ in range(1 << k)]


def make_logic(inputs, mode, rng):
    return [nested_canalizing_tt(len(ins), rng) if mode == "canalizing" else rand_tt(len(ins), rng)
            for ins in inputs]


def flat_features(inputs, tt, n):
    """ALL logic bits: per-node padded truth table + signed adjacency + summary."""
    v = np.zeros(NMAX * TTLEN + NMAX * NMAX, np.float32)
    for i in range(n):
        v[i * TTLEN:i * TTLEN + len(tt[i])] = np.array(tt[i], np.float32)
        for b, j in enumerate(inputs[i]):
            v[NMAX * TTLEN + i * NMAX + j] = edge_sign(inputs, tt, i, b)
    return np.concatenate([v, np.array(features(inputs, tt, n), np.float32)])


def async_reachable(inputs, tt, n, s0, target):
    seen = {s0}; dq = deque([s0])
    while dq:
        s = dq.popleft()
        if s == target:
            return 1
        for i in range(n):
            if node_next(s, i, inputs, tt) != ((s >> i) & 1):
                ns = s ^ (1 << i)
                if ns not in seen:
                    seen.add(ns); dq.append(ns)
    return int(target in seen)


def build_families(task, n_list, K_list, graphs_per_cell, v_each, seed):
    """Each graph: v_each random + v_each canalizing logic variants. Labels per task."""
    rng = random.Random(seed)
    Xs, Xf, y, gid, logic, summ = [], [], [], [], [], []
    g = 0
    for n in n_list:
        s0, target = 0, (1 << n) - 1            # async: all-zeros -> all-ones
        for K in K_list:
            for _ in range(graphs_per_cell):
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        tt = make_logic(inputs, mode, rng)
                        if task == "cyc":
                            lab = int(any(len(a) > 1 for a in sync_attractors(inputs, tt, n)))
                        else:
                            lab = async_reachable(inputs, tt, n, s0, target)
                        Xf.append(flat_features(inputs, tt, n))
                        summ.append(features(inputs, tt, n))
                        y.append(lab); gid.append(g); logic.append(mode)
                g += 1
    return (np.array(Xf, np.float32), np.array(summ, np.float32), np.array(y),
            np.array(gid), np.array(logic))


def deconf_split(gid, logic, seed=0):
    """Per graph: half of EACH logic type -> train, half -> test (balanced)."""
    rs = np.random.RandomState(seed); tr, te = [], []
    for gg in np.unique(gid):
        for mode in ("random", "canalizing"):
            ix = np.where((gid == gg) & (logic == mode))[0]
            ix = rs.permutation(ix); h = len(ix) // 2
            tr += list(ix[:h]); te += list(ix[h:])
    return np.array(tr), np.array(te)


def lgo_split(gid, seed=0):
    rs = np.random.RandomState(seed); g = rs.permutation(np.unique(gid))
    trg = set(g[:int(.7*len(g))])
    tr = np.array([i for i in range(len(gid)) if gid[i] in trg])
    te = np.array([i for i in range(len(gid)) if gid[i] not in trg])
    return tr, te


def flip_rate(y, gid, logic, mode=None):
    flips = tot = 0
    for gg in np.unique(gid):
        m = (gid == gg) if mode is None else ((gid == gg) & (logic == mode))
        labs = y[m]
        if len(labs) > 1:
            tot += 1; flips += int(len(set(labs)) > 1)
    return flips / max(tot, 1)


def iid_null_flip(y, gid):
    """P(>=1 flip among v i.i.d. draws at the marginal base rate)."""
    p = y.mean(); v = int(np.median([np.sum(gid == g) for g in np.unique(gid)]))
    return 1 - p**v - (1 - p)**v


def run_task(task, n_list, K_list, gpc, v_each):
    Xf, Xs, y, gid, logic = build_families(task, n_list, K_list, gpc, v_each, seed=7)
    print(f"\n================ TASK: {task} ================", flush=True)
    print(f"  {len(y)} nets, {len(np.unique(gid))} fixed-wiring graphs, base rate {y.mean():.2f}", flush=True)

    fr_all = flip_rate(y, gid, logic)
    fr_can = flip_rate(y, gid, logic, "canalizing")
    fr_rand = flip_rate(y, gid, logic, "random")
    print(f"  flip rate (label changes across logic on fixed wiring): all {100*fr_all:.0f}% | "
          f"random {100*fr_rand:.0f}% | canalizing {100*fr_can:.0f}% | i.i.d.-null {100*iid_null_flip(y,gid):.0f}%", flush=True)

    tr, te = deconf_split(gid, logic)
    # de-confounded within-graph: summary vs flattened-truth-table
    c1 = HistGradientBoostingClassifier(max_iter=300).fit(Xs[tr], y[tr])
    a_sum = auc(y[te], c1.predict_proba(Xs[te])[:, 1])
    c2 = HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=63).fit(Xf[tr], y[tr])
    a_flat = auc(y[te], c2.predict_proba(Xf[te])[:, 1])
    # shuffle null
    rs = np.random.RandomState(1); ysh = rs.permutation(y[tr])
    c3 = HistGradientBoostingClassifier(max_iter=300).fit(Xf[tr], ysh)
    a_null = auc(y[te], c3.predict_proba(Xf[te])[:, 1])
    print(f"  DE-CONFOUNDED within-graph AUROC: summary {a_sum:.3f} | FLATTENED-TT {a_flat:.3f} | shuffle-null {a_null:.3f}", flush=True)

    # leave-graph-out
    trg, teg = lgo_split(gid)
    c4 = HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=63).fit(Xf[trg], y[trg])
    a_lgo = auc(y[teg], c4.predict_proba(Xf[teg])[:, 1])
    print(f"  leave-graph-out AUROC (flattened-TT): {a_lgo:.3f}", flush=True)
    return task, a_flat, a_sum, a_null, fr_can


def main():
    res = []
    res.append(run_task("cyc", [12, 14], [2, 3, 4], 80, 4))
    res.append(run_task("reach", [10, 12], [2, 3, 4], 80, 4))
    print("\n================ DECISIVE GATE VERDICT ================", flush=True)
    for task, a_flat, a_sum, a_null, fr_can in res:
        tag = ("KILL" if a_flat <= 0.63 else "GO" if a_flat >= 0.75 else "PIVOT")
        sig = "ABOVE" if a_flat - a_null > 0.05 else "~= "
        print(f"  [{task}] flattened-TT within-graph AUROC {a_flat:.3f} ({sig} shuffle-null {a_null:.3f}); "
              f"canalizing flip {100*fr_can:.0f}% -> {tag}", flush=True)
    print("  Rule: KILL<=0.63 (all logic bits present, still stuck => unlearnable); GO>=0.75; else PIVOT.", flush=True)
    print("  If BOTH tasks KILL and canalizing flip is low -> AttractorNet's logic-wedge is dead; do NOT build the GNN.", flush=True)


if __name__ == "__main__":
    main()
