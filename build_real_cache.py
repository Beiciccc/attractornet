"""Stage A (robust, run once): fetch + fix-inputs + exact-label + featurize real BBM
nets, with a PER-NET wall-clock timeout (SIGALRM) so one pathological BDD blow-up
can't kill the whole run, INCREMENTAL pickle save, and per-net progress printing.

Output: real_cache.pkl = list of dicts {id, n, X(node feats), src, dst, sgn, y, base}.
Then Stage B (eval_transfer.py) loads this and runs synthetic-train + transfer eval.
"""
import argparse
import pickle
import signal
import time
import numpy as np
import biodivine_aeon as ba
from gnn_sparse import to_sparse
from day3_realtransfer import fix_inputs, real_net_to_repr, real_pernode_labels


class Timeout(Exception):
    pass


def _handler(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=70)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--out", default="real_cache.pkl")
    a = ap.parse_args()
    ids = ba.BiodivineBooleanModels.fetch_ids()
    cache, skip = [], {"size": 0, "param": 0, "timeout": 0, "err": 0, "novar": 0}
    t_start = time.time()
    for k, i in enumerate(ids):
        try:
            bn = ba.BiodivineBooleanModels.fetch_network(i)
            n = bn.variable_count()
            if n < 6 or n > a.max_n:
                skip["size"] += 1; continue
            fix_inputs(bn)
            signal.alarm(a.timeout)
            try:
                rep = real_net_to_repr(bn)
                if rep is None:
                    signal.alarm(0); skip["param"] += 1; continue
                inputs, tt = rep
                y = real_pernode_labels(bn)
            except Timeout:
                signal.alarm(0); skip["timeout"] += 1
                print(f"  [{k+1}/{len(ids)}] id={i} n={n} TIMEOUT", flush=True); continue
            finally:
                signal.alarm(0)
            X, src, dst, sgn = to_sparse(inputs, tt, n)
            rec = {"id": i, "n": n, "X": X, "src": src, "dst": dst, "sgn": sgn,
                   "y": y, "base": float(y.mean()), "both": int(len(np.unique(y)) > 1)}
            cache.append(rec)
            print(f"  [{k+1}/{len(ids)}] id={i} n={n} nodes={len(y)} base={y.mean():.2f} "
                  f"both={rec['both']}  (kept {len(cache)})", flush=True)
            if len(cache) % 10 == 0:
                pickle.dump(cache, open(a.out, "wb"))
        except Exception as e:
            skip["err"] += 1
            continue
    pickle.dump(cache, open(a.out, "wb"))
    nodes = sum(len(r["y"]) for r in cache)
    both = sum(r["both"] for r in cache)
    print(f"\nDONE in {time.time()-t_start:.0f}s. kept {len(cache)} nets ({nodes} nodes, "
          f"{both} nets with both classes). skips={skip}. saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
