"""Option B rescue — cheap probe of the SIGNAL-CONTROLLABILITY task BEFORE building a GNN.

Per node v: controllable = 1 iff v=1 is reachable from internal-quiescence (all
internal nodes OFF, inputs FREE) for SOME but NOT ALL input configurations. Genes
that always activate (regardless of signal) -> 0; never activate -> 0; activation
DEPENDS on the input -> 1. This needs the full COLORED forward reachability (the
EXPENSIVE oracle regime), and is biologically meaningful (signal-switchable genes).

This probe tests the two preconditions for the AAAI-rescue, per real BBM net:
  (1) is the exact oracle EXPENSIVE / does reach_fwd time grow & blow up with n?
  (2) is the controllability base rate NON-DEGENERATE (spread, not 0 or 1)?
Process-isolated with a hard timeout. If both hold, the amortization payoff exists.
"""
import argparse
import pickle
import time
import multiprocessing as mp
import numpy as np


def worker(i, q):
    try:
        import biodivine_aeon as ba
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        names = bn.variable_names()
        internal = [nm for nm, v in zip(names, bn.variables())
                    if not (bn.get_update_function(v) is None and len(bn.predecessors(v)) == 0)]
        n_inp = len(names) - len(internal)
        if n_inp == 0:                       # no free inputs -> controllability undefined
            q.put(("noinput", i, len(names))); return
        g = ba.AsynchronousGraph(bn)
        t0 = time.time()
        init = g.mk_subspace({nm: False for nm in internal})
        reached = ba.Reachability.reach_fwd(g, init)
        t_reach = time.time() - t0
        unit = g.mk_unit_colors().cardinality()
        on = []
        for nm in names:
            sub = g.mk_subspace_vertices({nm: True})
            cols = reached.intersect_vertices(sub).colors().cardinality()
            on.append(int(0 < cols < unit))
        y = np.array(on, np.float32)
        q.put(("ok", {"id": i, "n": len(names), "n_inputs": n_inp, "t_reach": t_reach,
                      "unit_colors": float(unit), "base": float(y.mean())}))
    except Exception as e:
        q.put(("err", i, repr(e)[:70]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=90)
    ap.add_argument("--max-n", type=int, default=400)
    a = ap.parse_args()
    import biodivine_aeon as ba
    ids = ba.BiodivineBooleanModels.fetch_ids()
    ctx = mp.get_context("spawn")
    rows, skip = [], {"size": 0, "noinput": 0, "timeout": 0, "err": 0}
    for k, i in enumerate(ids):
        try:
            n = ba.BiodivineBooleanModels.fetch_network(i).variable_count()
        except Exception:
            skip["err"] += 1; continue
        if n < 6 or n > a.max_n:
            skip["size"] += 1; continue
        q = ctx.Queue(); p = ctx.Process(target=worker, args=(i, q)); p.start(); p.join(a.budget)
        if p.is_alive():
            p.terminate(); p.join(); skip["timeout"] += 1
            rows.append({"id": i, "n": n, "n_inputs": None, "t_reach": None, "base": None})
            print(f"  id={i} n={n:>4} | reach_fwd TIMEOUT (>{a.budget}s)  <-- ORACLE EXPENSIVE", flush=True)
            continue
        try:
            res = q.get_nowait()
        except Exception:
            skip["err"] += 1; continue
        if res[0] == "ok":
            r = res[1]; rows.append(r)
            print(f"  id={i} n={r['n']:>4} inp={r['n_inputs']:>2} colors={r['unit_colors']:.3g} | "
                  f"reach {r['t_reach']:7.2f}s | controllable base {r['base']:.2f}", flush=True)
        else:
            skip[res[0] if res[0] in skip else "err"] += 1
    pickle.dump(rows, open("controllability_probe.pkl", "wb"))

    fin = [r for r in rows if r.get("t_reach") is not None]
    to = [r for r in rows if r.get("t_reach") is None and "n_inputs" in r and r["n_inputs"] is None]
    bases = [r["base"] for r in fin]
    times = [r["t_reach"] for r in fin]
    nondegen = [r for r in fin if 0.1 <= r["base"] <= 0.9]
    print("\n================ CONTROLLABILITY PROBE VERDICT ================", flush=True)
    print(f"  nets with free inputs evaluated: {len(fin)} | timeouts: {len(to)} | skips={skip}", flush=True)
    if fin:
        print(f"  reach_fwd time: median {np.median(times):.2f}s, p90 {np.percentile(times,90):.2f}s, max {np.max(times):.2f}s "
              f"(inputs-OFF was ~0.00s) -> (1) ORACLE EXPENSIVE?", flush=True)
        print(f"  controllability base: median {np.median(bases):.2f}, "
              f"{len(nondegen)}/{len(fin)} nets in [0.1,0.9] -> (2) NON-DEGENERATE?", flush=True)
        # cost vs n
        big = [r for r in fin if r["n"] >= 60]
        if big:
            print(f"  large nets (n>=60): reach times {sorted(round(r['t_reach'],1) for r in big)[-6:]}", flush=True)
    print(f"  GO for the GNN build iff oracle is clearly expensive (sec-to-timeout, grows with n) AND many nets non-degenerate.", flush=True)


if __name__ == "__main__":
    main()
