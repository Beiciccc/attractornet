"""Real-net cache for the FREE-INPUT COLORED reachability task (process-isolated).
Do NOT fix inputs: keep them free (aeon colors). Label per node v = 1 if v=1 is
reachable for SOME input configuration from internal quiescence (reach_fwd over the
colored async graph from {internal vars = false}; v activatable in some color).
Features use is_input-augmented arity-invariant descriptors (colored_task.to_sparse_ci).
"""
import argparse
import pickle
import time
import multiprocessing as mp
import numpy as np


def real_repr_ci(bn):
    """(inputs, tt, mask): input node (None fn, 0 regulators) -> mask=1, tt=None; else eval table."""
    from day3_realtransfer import eval_fn
    vars_ = bn.variables(); idx = {v: i for i, v in enumerate(vars_)}
    inputs, tt, mask = [], [], []
    for v in vars_:
        fn = bn.get_update_function(v); preds = bn.predecessors(v)
        if fn is None:
            if len(preds) == 0:
                inputs.append([]); tt.append(None); mask.append(1); continue
            return None                                   # implicit param over regulators
        supp = sorted(fn.support_variables(), key=lambda x: idx[x]); k = len(supp)
        if k > 18:
            return None
        try:
            table = [int(eval_fn(fn, {supp[b]: bool((x >> b) & 1) for b in range(k)})) for x in range(1 << k)]
        except ValueError:
            return None
        inputs.append([idx[s] for s in supp]); tt.append(table); mask.append(0)
    return inputs, tt, np.array(mask, np.int8)


def colored_labels(bn):
    """Per node v: 1 if v=1 reachable for SOME input config from internal-OFF (colored reach)."""
    import biodivine_aeon as ba
    g = ba.AsynchronousGraph(bn)
    names = bn.variable_names()
    # internal nodes = those with an update function; inputs (None fn, no regs) stay free
    internal = {nm for nm, v in zip(names, bn.variables())
                if not (bn.get_update_function(v) is None and len(bn.predecessors(v)) == 0)}
    init = g.mk_subspace({nm: False for nm in internal})  # internal off, inputs free -> colored
    reached = ba.Reachability.reach_fwd(g, init)
    on = []
    for nm in names:
        sub = g.mk_subspace_vertices({nm: True})
        on.append(int(not reached.intersect_vertices(sub).is_empty()))
    return np.array(on, np.float32)


def work(i, q):
    try:
        import biodivine_aeon as ba
        from colored_task import to_sparse_ci
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        n = bn.variable_count()
        rep = real_repr_ci(bn)
        if rep is None:
            q.put(("param", i, n)); return
        inputs, tt, mask = rep
        y = colored_labels(bn)
        X, src, dst, sgn = to_sparse_ci(inputs, tt, mask, n)
        q.put(("ok", {"id": i, "n": n, "X": X, "src": src, "dst": dst, "sgn": sgn,
                      "y": y, "base": float(y.mean()), "both": int(len(np.unique(y)) > 1),
                      "n_inputs": int(mask.sum())}))
    except Exception as e:
        q.put(("err", i, repr(e)[:70]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=70)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--out", default="real_cache_colored.pkl")
    ap.add_argument("--limit", type=int, default=10000)
    a = ap.parse_args()
    import biodivine_aeon as ba
    ids = ba.BiodivineBooleanModels.fetch_ids()
    ctx = mp.get_context("spawn")
    cache, skip = [], {"size": 0, "param": 0, "timeout": 0, "err": 0}
    t0 = time.time(); done = 0
    for k, i in enumerate(ids):
        if done >= a.limit:
            break
        try:
            n = ba.BiodivineBooleanModels.fetch_network(i).variable_count()
        except Exception:
            skip["err"] += 1; continue
        if n < 6 or n > a.max_n:
            skip["size"] += 1; continue
        done += 1
        q = ctx.Queue(); p = ctx.Process(target=work, args=(i, q)); p.start(); p.join(a.timeout)
        if p.is_alive():
            p.terminate(); p.join(); skip["timeout"] += 1
            print(f"  [{k+1}] id={i} n={n} TIMEOUT(killed)", flush=True); continue
        try:
            res = q.get_nowait()
        except Exception:
            skip["err"] += 1; continue
        if res[0] == "ok":
            r = res[1]; cache.append(r)
            print(f"  [{k+1}] id={i} n={r['n']} inp={r['n_inputs']} nodes={len(r['y'])} "
                  f"base={r['base']:.2f} both={r['both']}  (kept {len(cache)}, both1={sum(x['both'] for x in cache)})", flush=True)
            if len(cache) % 10 == 0:
                pickle.dump(cache, open(a.out, "wb"))
        else:
            skip[res[0] if res[0] in skip else "err"] += 1
    pickle.dump(cache, open(a.out, "wb"))
    both = sum(r["both"] for r in cache)
    print(f"\nDONE {time.time()-t0:.0f}s. kept {len(cache)} ({sum(len(r['y']) for r in cache)} nodes, "
          f"{both} both-class). skips={skip}. -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
