"""Robust, resumable, progress-printing labeler for real BBM nets.
Decoupled from training so a kill never loses work. Caches processed nets to
real_cache.pkl (list of {id,n,X,src,dst,sgn,y}); re-running resumes (skips cached).

For each net: fix external-input nodes to OFF, convert to (inputs, truth-table) repr,
compute per-node activation reachability via aeon symbolic reach_fwd(all-OFF) ∩ {v=1}.
Skips parameterized/high-arity/over-size nets and logs the reason.
"""
import argparse
import os
import pickle
import time
import numpy as np
import biodivine_aeon as ba
from gnn_sparse import to_sparse
from day3_realtransfer import fix_inputs, real_net_to_repr, real_pernode_labels

CACHE = "real_cache.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=72)
    ap.add_argument("--min-n", type=int, default=6)
    a = ap.parse_args()

    done = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            for rec in pickle.load(f):
                done[rec["id"]] = rec
        print(f"[resume] {len(done)} nets already cached", flush=True)

    ids = ba.BiodivineBooleanModels.fetch_ids()
    recs = list(done.values())
    skip = {"param": 0, "size": 0, "err": 0}
    for k, i in enumerate(ids):
        if i in done:
            continue
        try:
            bn = ba.BiodivineBooleanModels.fetch_network(i)
            n = bn.variable_count()
            if n < a.min_n or n > a.max_n:
                skip["size"] += 1; continue
            fix_inputs(bn)
            rep = real_net_to_repr(bn)
            if rep is None:
                skip["param"] += 1; continue
            inputs, tt = rep
            t0 = time.time()
            y = real_pernode_labels(bn)
            X, src, dst, sgn = to_sparse(inputs, tt, n)
            rec = {"id": i, "n": n, "X": X, "src": src, "dst": dst, "sgn": sgn,
                   "y": np.asarray(y, np.float32)}
            recs.append(rec)
            with open(CACHE, "wb") as f:           # checkpoint after every success
                pickle.dump(recs, f)
            print(f"[{k+1}/{len(ids)}] id={i} n={n} base={y.mean():.2f} "
                  f"label_t={time.time()-t0:.1f}s  (cached {len(recs)})", flush=True)
        except Exception as e:
            skip["err"] += 1
            print(f"[{k+1}/{len(ids)}] id={i} ERR {repr(e)[:70]}", flush=True)
    print(f"\nDONE: cached {len(recs)} real nets; skipped {skip}", flush=True)
    bothclass = sum(1 for r in recs if len(np.unique(r["y"])) > 1)
    print(f"nets with both classes: {bothclass}; total nodes: {sum(len(r['y']) for r in recs)}", flush=True)


if __name__ == "__main__":
    main()
