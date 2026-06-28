"""AttractorNet Day-1 GO/NO-GO (self-contained, CPU-only, no heavy deps).

Make-or-break question: does a GBDT on hand-crafted STRUCTURAL features (signed
feedback-loop counts a la Thomas's rules, canalizing depth, in-degree, size...)
SATURATE on Boolean-network attractor properties, or is there a non-trivial
LEARNABILITY / DIFFICULTY MAP (regions/knobs where even strong structural
features can't predict the exact-oracle label)?

We use an EXACT oracle (exhaustive state-transition-graph analysis on small nets:
sync attractors = cycles of the functional graph; async attractors = terminal
SCCs) so labels are ground truth, model-strength-immune — no weak-baseline gap.

GO   : a clean learnability gradient across the difficulty knob (K / size) and/or
       a size-generalization drop => the difficulty map has content (then a GNN
       can be tested against the structural-feature GBDT).
NO-GO: GBDT saturates (~perfect) across ALL K and sizes with no interpretable
       boundary => "cycle counts explain everything" => flat map, no contribution.
"""
import argparse
import itertools
import random
import numpy as np
import networkx as nx
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score


# --------------------------------------------------------------------------- #
# Boolean network: node i has inputs[i] (indices) and tt[i] (truth table bits) #
# --------------------------------------------------------------------------- #
def gen_bn(n, K, mode, rng):
    inputs, tt = [], []
    for i in range(n):
        k = min(K, n)
        ins = rng.sample(range(n), k)
        if mode == "canalizing":            # nested-canalizing: simpler dynamics
            table = nested_canalizing_tt(k, rng)
        else:                                # random (biased coin for output)
            p = rng.uniform(0.3, 0.7)
            table = [1 if rng.random() < p else 0 for _ in range(1 << k)]
        inputs.append(ins); tt.append(table)
    return inputs, tt


def nested_canalizing_tt(k, rng):
    """Build a nested-canalizing function's truth table over k inputs."""
    order = rng.sample(range(k), k)
    canal_in = [rng.randint(0, 1) for _ in range(k)]   # canalizing input value
    canal_out = [rng.randint(0, 1) for _ in range(k)]  # canalized output
    default = rng.randint(0, 1)
    table = []
    for x in range(1 << k):
        bits = [(x >> j) & 1 for j in range(k)]
        out = default
        for pos in range(k):
            j = order[pos]
            if bits[j] == canal_in[j]:
                out = canal_out[pos]; break
        table.append(out)
    return table


def step_sync(state, inputs, tt, n):
    ns = 0
    for i in range(n):
        idx = 0
        for b, j in enumerate(inputs[i]):
            idx |= ((state >> j) & 1) << b
        if tt[i][idx]:
            ns |= (1 << i)
    return ns


def node_next(state, i, inputs, tt):
    idx = 0
    for b, j in enumerate(inputs[i]):
        idx |= ((state >> j) & 1) << b
    return tt[i][idx]


# --------------------------------------------------------------------------- #
# EXACT oracle via exhaustive STG (small n)                                    #
# --------------------------------------------------------------------------- #
def sync_attractors(inputs, tt, n):
    N = 1 << n
    succ = [step_sync(s, inputs, tt, n) for s in range(N)]
    color = [0] * N                  # 0 unseen, 1 in-progress, 2 done
    attractors = []
    for s0 in range(N):
        if color[s0]:
            continue
        path = []
        s = s0
        while color[s] == 0:
            color[s] = 1; path.append(s); s = succ[s]
        if color[s] == 1:            # found a new cycle starting at s
            cyc, t = [], s
            while True:
                cyc.append(t); t = succ[t]
                if t == s:
                    break
            attractors.append(cyc)
        for v in path:
            color[v] = 2
    return attractors                # list of cycles (period = len)


def async_num_attractors(inputs, tt, n, cap=14):
    """Async attractors = terminal SCCs of the async STG (capped n)."""
    if n > cap:
        return None
    N = 1 << n
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for s in range(N):
        stable = True
        for i in range(n):
            if node_next(s, i, inputs, tt) != ((s >> i) & 1):
                G.add_edge(s, s ^ (1 << i)); stable = False
        if stable:
            G.add_edge(s, s)         # self-loop = async fixed point
    n_term = 0
    for scc in nx.strongly_connected_components(G):
        scc = set(scc)
        terminal = all(t in scc for u in scc for t in G.successors(u))
        if terminal:
            n_term += 1
    return n_term


# --------------------------------------------------------------------------- #
# hand-crafted STRUCTURAL features (the strong baseline to beat/saturate)      #
# --------------------------------------------------------------------------- #
def edge_sign(inputs, tt, i, b):
    """Sign of input b of node i: monotone increasing(+1)/decreasing(-1)/mixed(0)."""
    k = len(inputs[i]); inc = dec = True
    for x in range(1 << k):
        if (x >> b) & 1:
            continue
        y = x | (1 << b)
        if tt[i][y] < tt[i][x]: inc = False
        if tt[i][y] > tt[i][x]: dec = False
    return 1 if inc and not dec else (-1 if dec and not inc else 0)


def is_canalizing_depth(table, k):
    """Return canalizing depth (# of nested canalizing layers)."""
    depth = 0; idxs = list(range(1 << k)); active = list(range(k))
    cur = table[:]
    # crude: count inputs that are canalizing at top level (depth approx)
    for b in active:
        vals0 = [cur[x] for x in range(len(cur)) if not ((x >> b) & 1)]
        vals1 = [cur[x] for x in range(len(cur)) if ((x >> b) & 1)]
        if len(set(vals0)) == 1 or len(set(vals1)) == 1:
            depth += 1
    return depth


def features(inputs, tt, n):
    G = nx.DiGraph(); G.add_nodes_from(range(n))
    signs = {}
    for i in range(n):
        for b, j in enumerate(inputs[i]):
            s = edge_sign(inputs, tt, i, b)
            G.add_edge(j, i); signs[(j, i)] = s
    pos_cyc = neg_cyc = tot_cyc = 0
    try:
        for cyc in nx.simple_cycles(G, length_bound=4):
            prod = 1; ok = True
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                sgn = signs.get((a, b), 0)
                if sgn == 0: ok = False; break
                prod *= sgn
            tot_cyc += 1
            if ok and prod > 0: pos_cyc += 1
            elif ok and prod < 0: neg_cyc += 1
    except Exception:
        pass
    indeg = [len(inputs[i]) for i in range(n)]
    outdeg = [G.out_degree(i) for i in range(n)]
    canal = [is_canalizing_depth(tt[i], len(inputs[i])) for i in range(n)]
    bias = [sum(tt[i]) / len(tt[i]) for i in range(n)]
    return [
        n, np.mean(indeg), np.max(indeg), np.mean(outdeg), np.max(outdeg),
        sum(1 for d in [G.in_degree(i) for i in range(n)] if d == 0),     # sources
        sum(1 for d in outdeg if d == 0),                                  # sinks
        pos_cyc, neg_cyc, tot_cyc,
        np.mean(canal), np.max(canal),
        np.mean(bias), np.std(bias),
    ]


# --------------------------------------------------------------------------- #
# build dataset, train GBDT, report learnability map                          #
# --------------------------------------------------------------------------- #
def build(n_list, K_list, modes, per_cell, seed):
    rng = random.Random(seed)
    X, y_cyc, y_cnt, K_of, n_of = [], [], [], [], []
    for n in n_list:
        for K in K_list:
            for mode in modes:
                for _ in range(per_cell):
                    inp, tt = gen_bn(n, K, mode, rng)
                    atts = sync_attractors(inp, tt, n)
                    X.append(features(inp, tt, n))
                    y_cyc.append(int(any(len(a) > 1 for a in atts)))   # has cyclic attractor
                    y_cnt.append(len(atts))                            # attractor count
                    K_of.append(K); n_of.append(n)
    return (np.array(X, float), np.array(y_cyc), np.array(y_cnt),
            np.array(K_of), np.array(n_of))


def auc_or_nan(yt, ys):
    return roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_list = [10, 12, 14, 16]
    K_list = [1, 2, 3, 4, 5]
    modes = ["random", "canalizing"]
    print(f"[build] n={n_list} K={K_list} modes={modes} per_cell={a.per_cell} "
          f"(total {len(n_list)*len(K_list)*len(modes)*a.per_cell} exact-labeled nets)")
    X, ycyc, ycnt, Kof, nof = build(n_list, K_list, modes, a.per_cell, a.seed)
    print(f"[build] done. has-cyclic base rate {ycyc.mean():.2f}; attractor-count mean {ycnt.mean():.2f}")

    # in-distribution split
    rng = np.random.RandomState(0); idx = rng.permutation(len(X))
    tr, te = idx[:int(.7*len(idx))], idx[int(.7*len(idx)):]

    clf = HistGradientBoostingClassifier(max_iter=300).fit(X[tr], ycyc[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc_overall = auc_or_nan(ycyc[te], p)
    reg = HistGradientBoostingRegressor(max_iter=300).fit(X[tr], np.log1p(ycnt[tr]))
    pr = reg.predict(X[te])
    r2_cnt = r2_score(np.log1p(ycnt[te]), pr)

    # NO-GO control: does a single Thomas-rule feature (neg-feedback-circuit count,
    # idx 8) already explain has-cyclic as well as the full GBDT? If full >> thomas,
    # there is learnable structure BEYOND cycle-counting. negate sign so more neg-cyc
    # -> higher cyclic-attractor score (Thomas: negative circuits drive oscillation).
    thomas = X[te][:, 8]
    auc_thomas = auc_or_nan(ycyc[te], thomas)

    print("\n=== Task A: has-cyclic-attractor (binary) — GBDT on structural features ===")
    print(f"  in-distribution AUROC: {auc_overall:.3f}   (Thomas neg-circuit-only baseline: {auc_thomas:.3f})")
    print(f"  headroom of full features beyond raw cycle-counting: {auc_overall-auc_thomas:+.3f}")
    print("  AUROC per difficulty knob K (in-degree):")
    for K in K_list:
        m = (Kof[te] == K)
        at = auc_or_nan(ycyc[te][m], thomas[m])
        print(f"    K={K}:  GBDT {auc_or_nan(ycyc[te][m], p[m]):.3f}  Thomas {at:.3f}"
              f"   (base rate {ycyc[te][m].mean():.2f}, n={m.sum()})")

    print("\n=== Task B: log attractor-count (regression) — GBDT R2 ===")
    print(f"  in-distribution R2: {r2_cnt:.3f}")
    print("  R2 per difficulty knob K:")
    for K in K_list:
        m = (Kof[te] == K)
        if m.sum() > 5:
            print(f"    K={K}:  R2 {r2_score(np.log1p(ycnt[te][m]), pr[m]):.3f}"
                  f"   (mean count {ycnt[te][m].mean():.2f}, n={m.sum()})")

    # size-generalization: train n<=14, test n=16
    trs = nof <= 14; tes = nof == 16
    clf2 = HistGradientBoostingClassifier(max_iter=300).fit(X[trs], ycyc[trs])
    p2 = clf2.predict_proba(X[tes])[:, 1]
    auc_sizegen = auc_or_nan(ycyc[tes], p2)
    print("\n=== Size-generalization (train n<=14 -> test n=16), Task A ===")
    print(f"  AUROC: {auc_sizegen:.3f}  (vs in-dist {auc_overall:.3f}; a drop = non-trivial size structure)")

    # verdict
    per_K = [auc_or_nan(ycyc[te][Kof[te] == K], p[Kof[te] == K]) for K in K_list]
    per_K = [x for x in per_K if x == x]
    spread = (max(per_K) - min(per_K)) if per_K else 0.0
    saturated = (auc_overall > 0.97) and (min(per_K) > 0.95 if per_K else False)
    gradient = spread > 0.08 or (auc_overall - auc_sizegen) > 0.05
    print("\n================ DAY-1 GO/NO-GO VERDICT ================")
    print(f"  GBDT overall AUROC {auc_overall:.3f} | per-K spread {spread:.3f} | size-gen drop {auc_overall-auc_sizegen:+.3f}")
    if saturated and not gradient:
        print("  >>> NO-GO: structural-feature GBDT saturates across all K & sizes with no learnability")
        print("      gradient -> 'cycle counts explain everything' -> flat difficulty map, no content.")
    elif gradient:
        print("  >>> GO: a learnability GRADIENT exists (per-K spread and/or size-gen drop) -> the")
        print("      difficulty map has content. Next: test whether a GNN recovers the hard regions a")
        print("      structural-feature GBDT cannot, + cross-distribution transfer to real BBM nets.")
    else:
        print("  >>> BORDERLINE: no saturation but weak gradient — widen knobs (async task, larger K,")
        print("      harder canalizing mix) before committing.")


if __name__ == "__main__":
    main()
