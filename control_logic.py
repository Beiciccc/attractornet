"""Day-1.5b DECISIVE CONTROL for the SURVIVING WEDGE (post scoop-recon).

The scoop recon killed the 'difficulty map' headline (Maheshwari/Lesne/Hutt 2023
already built a structure-keyed predictability landscape). The ONLY non-scooped,
non-weak-baseline wedge it identified:

   A learned model that extracts LOGIC-DEPENDENT attractor information that NO
   interaction-graph feature can encode -- provable headroom because the same
   wiring diagram + different Boolean logic yields different attractors (counting
   fixed points is #P-complete; the interaction graph under-determines dynamics).

This script tests that wedge directly, two ways:

(1) FEATURE DECOMPOSITION: split features into
      topology-only  : computable from the UNSIGNED wiring alone (in/out-degree,
                       unsigned cycle counts, sources/sinks, size) -- constant
                       across logic variants of a fixed graph.
      signed-graph   : + Thomas signed-circuit counts (pos/neg) -- the STRONG
                       interaction-graph baseline (Thomas/Aracena level).
      logic-aware    : + canalizing depth, output bias, node sensitivity -- sees
                       the actual Boolean functions.
    If logic-aware >> signed-graph on attractor count/type, the logic carries
    predictive signal beyond the (signed) interaction graph.

(2) SAME-GRAPH / DIFFERENT-LOGIC PAIRED CONTROL (the killer): hold wiring FIXED,
    vary only the Boolean functions. Measure the fraction of graphs whose
    has-cyclic label FLIPS across logic variants (= provable headroom no topology
    feature can capture). Then a within-graph split (train on some variants, test
    on held-out variants of the SAME graphs) forces any topology-only model to
    chance, and we ask whether a logic-aware model still predicts. If yes -> the
    wedge is real and information-theoretically guaranteed, NOT a weak-baseline trap.
"""
import random
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from attractor_gonogo import (gen_bn, nested_canalizing_tt, sync_attractors,
                              features, step_sync)

TOPO = [0, 1, 2, 3, 4, 5, 6, 9]            # unsigned wiring only
SIGNED = TOPO + [7, 8]                      # + Thomas signed circuits
LOGIC = SIGNED + [10, 11, 12, 13]           # + canalizing / bias (Boolean-function info)


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def fit_auc(Xtr, ytr, Xte, yte, cols):
    if len(np.unique(ytr)) < 2:
        return float("nan")
    c = HistGradientBoostingClassifier(max_iter=250).fit(Xtr[:, cols], ytr)
    return auc(yte, c.predict_proba(Xte[:, cols])[:, 1])


def fit_r2(Xtr, ytr, Xte, yte, cols):
    r = HistGradientBoostingRegressor(max_iter=250).fit(Xtr[:, cols], np.log1p(ytr))
    return r2_score(np.log1p(yte), r.predict(Xte[:, cols]))


def random_tt(k, rng):
    p = rng.uniform(0.3, 0.7)
    return [1 if rng.random() < p else 0 for _ in range(1 << k)]


def relabel_logic(inputs, mode, rng):
    """Fresh Boolean functions on a FIXED wiring `inputs`."""
    tt = []
    for ins in inputs:
        k = len(ins)
        tt.append(nested_canalizing_tt(k, rng) if mode == "canalizing" else random_tt(k, rng))
    return tt


# --------------------------------------------------------------------------- #
# Part 1: feature decomposition on the standard ensemble                       #
# --------------------------------------------------------------------------- #
def part1():
    rng = random.Random(0)
    n_list, K_list, modes, PER = [10, 12, 14, 16], [1, 2, 3, 4, 5], ["random", "canalizing"], 120
    X, ycyc, ycnt, Kof = [], [], [], []
    for n in n_list:
        for K in K_list:
            for mode in modes:
                for _ in range(PER):
                    inp, tt = gen_bn(n, K, mode, rng)
                    atts = sync_attractors(inp, tt, n)
                    X.append(features(inp, tt, n))
                    ycyc.append(int(any(len(a) > 1 for a in atts)))
                    ycnt.append(len(atts)); Kof.append(K)
    X = np.array(X, float); ycyc = np.array(ycyc); ycnt = np.array(ycnt); Kof = np.array(Kof)
    rs = np.random.RandomState(0); idx = rs.permutation(len(X))
    tr, te = idx[:int(.7*len(idx))], idx[int(.7*len(idx)):]
    print("=== Part 1: feature-group decomposition (topology vs signed-graph vs logic-aware) ===", flush=True)
    print(f"  {'group':<14}{'has-cyclic AUROC':>18}{'count R2':>12}", flush=True)
    for name, cols in [("topology", TOPO), ("signed-graph", SIGNED), ("logic-aware", LOGIC)]:
        print(f"  {name:<14}{fit_auc(X[tr],ycyc[tr],X[te],ycyc[te],cols):>18.3f}{fit_r2(X[tr],ycnt[tr],X[te],ycnt[te],cols):>12.3f}", flush=True)
    print("  logic-gain = logic-aware minus signed-graph (the wedge):", flush=True)
    g_auc = fit_auc(X[tr],ycyc[tr],X[te],ycyc[te],LOGIC)-fit_auc(X[tr],ycyc[tr],X[te],ycyc[te],SIGNED)
    g_r2 = fit_r2(X[tr],ycnt[tr],X[te],ycnt[te],LOGIC)-fit_r2(X[tr],ycnt[tr],X[te],ycnt[te],SIGNED)
    print(f"    has-cyclic AUROC +{g_auc:.3f} | count R2 +{g_r2:.3f}", flush=True)
    return g_auc, g_r2


# --------------------------------------------------------------------------- #
# Part 2: same-graph / different-logic paired control                          #
# --------------------------------------------------------------------------- #
def part2():
    rng = random.Random(7)
    n_list, K_list = [12, 14], [2, 3, 4]
    GRAPHS_PER_CELL, VARIANTS = 80, 8
    X, ycyc, ycnt, gid = [], [], [], []
    g = 0
    flip_total = flip_cyclic = 0
    for n in n_list:
        for K in K_list:
            for _ in range(GRAPHS_PER_CELL):
                # fixed wiring for this graph
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                labels_here = []
                for v in range(VARIANTS):
                    mode = "random" if v % 2 == 0 else "canalizing"
                    tt = relabel_logic(inputs, mode, rng)
                    atts = sync_attractors(inputs, tt, n)
                    yc = int(any(len(a) > 1 for a in atts))
                    X.append(features(inputs, tt, n)); ycyc.append(yc)
                    ycnt.append(len(atts)); gid.append(g)
                    labels_here.append(yc)
                flip_total += 1
                if len(set(labels_here)) > 1:        # has-cyclic FLIPS within fixed wiring
                    flip_cyclic += 1
                g += 1
    X = np.array(X, float); ycyc = np.array(ycyc); ycnt = np.array(ycnt); gid = np.array(gid)

    print("\n=== Part 2: same-graph / different-logic paired control ===", flush=True)
    print(f"  graphs={flip_total}  variants/graph={VARIANTS}", flush=True)
    print(f"  has-cyclic label FLIPS across logic on the SAME wiring: "
          f"{flip_cyclic}/{flip_total} = {100*flip_cyclic/flip_total:.1f}% of graphs", flush=True)
    print("    -> on these graphs ANY topology-only feature is provably at chance (graph is constant).", flush=True)

    # within-graph split: train on variants of a graph subset's first half indices,
    # test on held-out VARIANTS of the SAME graphs -> topology features ~constant per graph.
    gids = np.unique(gid)
    # split variants, not graphs: for each graph put alternating variants in tr/te
    order_in_graph = np.zeros(len(gid), int)
    for gg in gids:
        m = np.where(gid == gg)[0]
        for r, ix in enumerate(m):
            order_in_graph[ix] = r
    tr = order_in_graph % 2 == 0
    te = order_in_graph % 2 == 1
    print("  within-graph split (same graphs in train & test, different logic variants):", flush=True)
    print(f"    {'group':<14}{'has-cyclic AUROC':>18}{'count R2':>12}", flush=True)
    for name, cols in [("topology", TOPO), ("signed-graph", SIGNED), ("logic-aware", LOGIC)]:
        a = fit_auc(X[tr], ycyc[tr], X[te], ycyc[te], cols)
        r2 = fit_r2(X[tr], ycnt[tr], X[te], ycnt[te], cols)
        print(f"    {name:<14}{a:>18.3f}{r2:>12.3f}", flush=True)
    return flip_cyclic / flip_total


def main():
    g_auc, g_r2 = part1()
    flip = part2()
    print("\n================ SURVIVING-WEDGE VERDICT ================", flush=True)
    print(f"  logic-gain over signed-graph: has-cyclic +{g_auc:.3f}, count R2 +{g_r2:.3f}", flush=True)
    print(f"  has-cyclic flips across logic on fixed wiring: {100*flip:.1f}% of graphs", flush=True)
    if flip > 0.15 and (g_auc > 0.04 or g_r2 > 0.05):
        print("  >>> WEDGE CONFIRMED: logic carries attractor signal the (signed) interaction graph", flush=True)
        print("      cannot, and a large fraction of graphs have logic-flippable labels => provable", flush=True)
        print("      headroom for a logic-aware learner (GNN-on-functions). This is the scoop-resistant,", flush=True)
        print("      non-weak-baseline contribution. GO to the GNN-vs-strong-features head-to-head.", flush=True)
    elif flip <= 0.05:
        print("  >>> WEAK: wiring nearly determines has-cyclic => little logic headroom for THIS task;", flush=True)
        print("      foreground attractor COUNT / async reachability instead (likely higher headroom).", flush=True)
    else:
        print("  >>> BORDERLINE: some headroom; let the adversarial panel weigh task choice & framing.", flush=True)


if __name__ == "__main__":
    main()
