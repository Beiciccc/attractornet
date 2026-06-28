"""Trivial-bias control for the async-reachability finding (theorist objection C):
is AUROC ~0.92 just 'mean output bias predicts all-ones reachability'?

Checks, on de-confounded within-graph fixed-wiring families:
  - target = ALL-ONES (the gate's target) vs target = a NON-TRIVIAL fixed state.
  - mean-bias-ALONE AUROC vs full summary vs flattened-TT.
If mean-bias-alone already ~= full, the finding is a trivial bias artifact for the
all-ones target; a non-trivial target should break that while keeping real signal.
"""
import random
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from attractor_gonogo import features
from decisive_gate import make_logic, flat_features, async_reachable, deconf_split


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def build(target_kind, n_list, K_list, gpc, v_each, seed=7):
    rng = random.Random(seed)
    Xf, Xs, y, gid, logic = [], [], [], [], []
    g = 0
    for n in n_list:
        s0 = 0
        target = (1 << n) - 1 if target_kind == "ones" else int("01" * n, 2) & ((1 << n) - 1)
        for K in K_list:
            for _ in range(gpc):
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        tt = make_logic(inputs, mode, rng)
                        y.append(async_reachable(inputs, tt, n, s0, target))
                        Xf.append(flat_features(inputs, tt, n)); Xs.append(features(inputs, tt, n))
                        gid.append(g); logic.append(mode)
                g += 1
    return (np.array(Xf, np.float32), np.array(Xs, np.float32), np.array(y),
            np.array(gid), np.array(logic, object))


def gb(X, y, tr, te, **kw):
    if len(np.unique(y[tr])) < 2:
        return float("nan")
    c = HistGradientBoostingClassifier(max_iter=400, **kw).fit(X[tr], y[tr])
    return auc(y[te], c.predict_proba(X[te])[:, 1])


def main():
    for tk in ("ones", "nontrivial"):
        Xf, Xs, y, gid, logic = build(tk, [10, 12], [2, 3, 4], 80, 4)
        tr, te = deconf_split(gid, logic)
        bias = Xs[:, 12:13]                      # mean output bias only
        a_bias = gb(bias, y, tr, te)
        a_sum = gb(Xs, y, tr, te)
        a_flat = gb(Xf, y, tr, te, max_leaf_nodes=63)
        print(f"[target={tk:<10}] base {y.mean():.2f} | mean-bias-ALONE {a_bias:.3f} | "
              f"summary {a_sum:.3f} | flat-TT {a_flat:.3f}  (flat-minus-bias {a_flat-a_bias:+.3f})", flush=True)
    print("  If flat-minus-bias is large (>0.1), reachability predictability is NOT a trivial bias artifact.", flush=True)


if __name__ == "__main__":
    main()
