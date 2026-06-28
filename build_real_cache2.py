"""Stage A (robust v2): PROCESS-ISOLATED per-net labeling with a HARD timeout.
SIGALRM cannot interrupt aeon's Rust BDD calls (reach_fwd); only killing the
process can. Each net is fetched+labeled+featurized in a child process; the parent
terminates any child that exceeds the wall clock. Incremental pickle save + per-net
progress. Output: real_cache.pkl.
"""
import argparse
import pickle
import time
import multiprocessing as mp
import numpy as np


def work(i, q):
    try:
        import biodivine_aeon as ba
        from gnn_sparse import to_sparse
        from day3_realtransfer import fix_inputs, real_net_to_repr, real_pernode_labels
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        n = bn.variable_count()
        fix_inputs(bn)
        rep = real_net_to_repr(bn)
        if rep is None:
            q.put(("param", i, n)); return
        inputs, tt = rep
        y = real_pernode_labels(bn)
        X, src, dst, sgn = to_sparse(inputs, tt, n)
        q.put(("ok", {"id": i, "n": n, "X": X, "src": src, "dst": dst, "sgn": sgn,
                      "y": y, "base": float(y.mean()), "both": int(len(np.unique(y)) > 1)}))
    except Exception as e:
        q.put(("err", i, repr(e)[:60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=70)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", default="real_cache.pkl")
    a = ap.parse_args()
    import biodivine_aeon as ba
    ids = ba.BiodivineBooleanModels.fetch_ids()
    ctx = mp.get_context("spawn")
    cache, skip = [], {"size": 0, "param": 0, "timeout": 0, "err": 0}
    t0 = time.time()
    for k, i in enumerate(ids):
        try:
            bn = ba.BiodivineBooleanModels.fetch_network(i)
            n = bn.variable_count()
        except Exception:
            skip["err"] += 1; continue
        if n < 6 or n > a.max_n:
            skip["size"] += 1; continue
        q = ctx.Queue()
        p = ctx.Process(target=work, args=(i, q))
        p.start(); p.join(a.timeout)
        if p.is_alive():
            p.terminate(); p.join(); skip["timeout"] += 1
            print(f"  [{k+1}/{len(ids)}] id={i} n={n} TIMEOUT(killed)", flush=True); continue
        try:
            res = q.get_nowait()
        except Exception:
            skip["err"] += 1; continue
        if res[0] == "ok":
            rec = res[1]; cache.append(rec)
            print(f"  [{k+1}/{len(ids)}] id={i} n={rec['n']} nodes={len(rec['y'])} "
                  f"base={rec['base']:.2f} both={rec['both']}  (kept {len(cache)}, both1={sum(r['both'] for r in cache)})", flush=True)
            if len(cache) % 10 == 0:
                pickle.dump(cache, open(a.out, "wb"))
        else:
            skip[res[0] if res[0] in skip else "err"] += 1
    pickle.dump(cache, open(a.out, "wb"))
    nodes = sum(len(r["y"]) for r in cache); both = sum(r["both"] for r in cache)
    print(f"\nDONE {time.time()-t0:.0f}s. kept {len(cache)} nets ({nodes} nodes, {both} both-class). "
          f"skips={skip}. -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
