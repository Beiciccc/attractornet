"""Day-1.5 DECISIVE CONTROL: is the learnability gradient just the textbook
Kauffman-Derrida order->chaos transition (scooped by 1986 theory), or is there
per-network structure-to-dynamics signal BEYOND the mean-field chaos order param?

The killer attacks on the Day-1 GO signal:
  (A) CHAOS-CONFOUND / NOVELTY-SCOOP: compute the empirical Derrida slope lambda
      (expected post-step Hamming distance from a distance-1 pair; >1 chaotic).
      If lambda ALONE explains has-cyclic / count-unpredictability as well as the
      full feature set, the difficulty map == the order-chaos transition (known).
  (B) BEYOND-MEAN-FIELD: does the full structural-feature GBDT beat lambda-alone
      per-network? If yes, there is per-net structure->dynamics signal that the
      mean-field theory misses -> a real ML contribution exists. If no, NO-GO.
  (C) base-rate / mode confound: break has-cyclic AUROC by mode (random vs
      canalizing) to see if the non-monotonic per-K curve is an imbalance artifact.
"""
import random
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from attractor_gonogo import gen_bn, sync_attractors, features, step_sync


def derrida_slope(inputs, tt, n, rng, samples=300):
    """Empirical Derrida coefficient: E[Hamming(F(s), F(s')) | Hamming(s,s')=1]."""
    tot = 0
    for _ in range(samples):
        s = rng.randrange(1 << n)
        b = rng.randrange(n)
        s2 = s ^ (1 << b)
        d = (step_sync(s, inputs, tt, n) ^ step_sync(s2, inputs, tt, n))
        tot += bin(d).count("1")
    return tot / samples


def build(n_list, K_list, modes, per_cell, seed):
    rng = random.Random(seed)
    drng = random.Random(seed + 999)
    X, lam, ycyc, ycnt, Kof, Mof = [], [], [], [], [], []
    for n in n_list:
        for K in K_list:
            for mi, mode in enumerate(modes):
                for _ in range(per_cell):
                    inp, tt = gen_bn(n, K, mode, rng)
                    atts = sync_attractors(inp, tt, n)
                    X.append(features(inp, tt, n))
                    lam.append(derrida_slope(inp, tt, n, drng))
                    ycyc.append(int(any(len(a) > 1 for a in atts)))
                    ycnt.append(len(atts)); Kof.append(K); Mof.append(mi)
    return (np.array(X, float), np.array(lam)[:, None], np.array(ycyc),
            np.array(ycnt), np.array(Kof), np.array(Mof))


def auc(yt, s):
    return roc_auc_score(yt, s) if len(np.unique(yt)) > 1 else float("nan")


def main():
    n_list, K_list, modes = [10, 12, 14, 16], [1, 2, 3, 4, 5], ["random", "canalizing"]
    PER = 120
    print(f"[build] {len(n_list)*len(K_list)*len(modes)*PER} nets + Derrida slope each", flush=True)
    X, lam, ycyc, ycnt, Kof, Mof = build(n_list, K_list, modes, PER, 0)
    Xfull = np.hstack([X, lam])                     # full features + Derrida
    print(f"[build] done. lambda range [{lam.min():.2f},{lam.max():.2f}] "
          f"mean {lam.mean():.2f}; has-cyclic base {ycyc.mean():.2f}", flush=True)

    rs = np.random.RandomState(0); idx = rs.permutation(len(X))
    tr, te = idx[:int(.7*len(idx))], idx[int(.7*len(idx)):]

    def fit_auc(Xtr, ytr, Xte, yte):
        c = HistGradientBoostingClassifier(max_iter=300).fit(Xtr, ytr)
        return auc(yte, c.predict_proba(Xte)[:, 1])

    def fit_r2(Xtr, ytr, Xte, yte):
        r = HistGradientBoostingRegressor(max_iter=300).fit(Xtr, np.log1p(ytr))
        return r2_score(np.log1p(yte), r.predict(Xte))

    # ---- has-cyclic: lambda-alone vs full-vs-(full+lambda) ----
    print("\n=== has-cyclic-attractor AUROC ===", flush=True)
    print(f"  lambda-ALONE (Derrida)      : {fit_auc(lam[tr], ycyc[tr], lam[te], ycyc[te]):.3f}")
    print(f"  structural features         : {fit_auc(X[tr], ycyc[tr], X[te], ycyc[te]):.3f}")
    print(f"  structural + lambda         : {fit_auc(Xfull[tr], ycyc[tr], Xfull[te], ycyc[te]):.3f}")

    # ---- count R2: lambda-alone vs full ----
    print("\n=== log attractor-count R2 ===", flush=True)
    print(f"  lambda-ALONE (Derrida)      : {fit_r2(lam[tr], ycnt[tr], lam[te], ycnt[te]):.3f}")
    print(f"  structural features         : {fit_r2(X[tr], ycnt[tr], X[te], ycnt[te]):.3f}")
    print(f"  structural + lambda         : {fit_r2(Xfull[tr], ycnt[tr], Xfull[te], ycnt[te]):.3f}")

    # ---- the decisive per-K table: does the gradient collapse onto lambda? ----
    print("\n=== per-K: is the difficulty gradient explained by lambda alone? ===", flush=True)
    print(f"  {'K':>2} {'lam_mean':>9} {'cyc_lamAUC':>11} {'cyc_fullAUC':>12} {'cnt_lamR2':>10} {'cnt_fullR2':>11} {'beyond':>8}", flush=True)
    cl = HistGradientBoostingClassifier(max_iter=300).fit(lam[tr], ycyc[tr]); pl = cl.predict_proba(lam[te])[:, 1]
    cf = HistGradientBoostingClassifier(max_iter=300).fit(X[tr], ycyc[tr]); pf = cf.predict_proba(X[te])[:, 1]
    rl = HistGradientBoostingRegressor(max_iter=300).fit(lam[tr], np.log1p(ycnt[tr])); ql = rl.predict(lam[te])
    rf = HistGradientBoostingRegressor(max_iter=300).fit(X[tr], np.log1p(ycnt[tr])); qf = rf.predict(X[te])
    beyond_vals = []
    for K in K_list:
        m = Kof[te] == K
        la, fa = auc(ycyc[te][m], pl[m]), auc(ycyc[te][m], pf[m])
        lr = r2_score(np.log1p(ycnt[te][m]), ql[m]); fr = r2_score(np.log1p(ycnt[te][m]), qf[m])
        beyond = (fa - la)  # how much full features add over lambda alone (has-cyclic)
        beyond_vals.append(beyond if beyond == beyond else 0)
        print(f"  {K:>2} {lam[te][m].mean():>9.2f} {la:>11.3f} {fa:>12.3f} {lr:>10.3f} {fr:>11.3f} {beyond:>+8.3f}", flush=True)

    # ---- mode breakdown (base-rate confound) ----
    print("\n=== has-cyclic AUROC by mode (base-rate/mode confound) ===", flush=True)
    for mi, mode in enumerate(modes):
        mm = Mof[te] == mi
        print(f"  {mode:<12} base {ycyc[te][mm].mean():.2f}  fullAUC {auc(ycyc[te][mm], pf[mm]):.3f}  lamAUC {auc(ycyc[te][mm], pl[mm]):.3f}")

    # ---- verdict ----
    cyc_full = fit_auc(X[tr], ycyc[tr], X[te], ycyc[te])
    cyc_lam = fit_auc(lam[tr], ycyc[tr], lam[te], ycyc[te])
    cnt_full = fit_r2(X[tr], ycnt[tr], X[te], ycnt[te])
    cnt_lam = fit_r2(lam[tr], ycnt[tr], lam[te], ycnt[te])
    beyond_cyc = cyc_full - cyc_lam
    beyond_cnt = cnt_full - cnt_lam
    print("\n================ CHAOS-CONFOUND VERDICT ================", flush=True)
    print(f"  beyond-lambda gain  has-cyclic AUROC {beyond_cyc:+.3f} | count R2 {beyond_cnt:+.3f}", flush=True)
    if beyond_cyc < 0.04 and beyond_cnt < 0.05:
        print("  >>> SCOOP/NO-GO: full structure adds ~nothing over the Derrida order param.")
        print("      The 'difficulty map' IS the textbook Kauffman order-chaos transition.")
    elif beyond_cyc > 0.08 or beyond_cnt > 0.10:
        print("  >>> SURVIVES: full per-network structure beats mean-field lambda substantially")
        print("      -> there is structure->dynamics signal the order-chaos theory misses. A GNN")
        print("      that beats this GBDT would be a genuine (non-scooped) contribution.")
    else:
        print("  >>> BORDERLINE: modest gain over lambda; needs the GNN-vs-GBDT test to decide.")


if __name__ == "__main__":
    main()
