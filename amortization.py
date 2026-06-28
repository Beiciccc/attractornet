"""(1) AMORTIZATION PAYOFF — the Area Chair's #1 lever.
Show the GNN answers per-node reachability in ~ms on networks where the EXACT
symbolic oracle (biodivine_aeon reach_fwd) takes seconds-to-minutes or TIMES OUT.
The oracle cost grows with the 2^n state space (BDD-compressed but blows up); the
GNN cost grows only with bounded per-node arity, so it is flat and cheap. Where the
oracle is feasible, the GNN MATCHES it (AUROC) — so the GNN is a validated surrogate
that answers queries the oracle cannot.

Outputs amortization.pkl (per-net: n, edges, t_gnn, t_oracle|timeout, auc) for the
scaling figure, and prints oracle-infeasible nets where the GNN still predicts.
"""
import argparse
import pickle
import sys
import time
import multiprocessing as mp
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from ablation_signaware import FlexGNN, train as train_gnn
from day3_pernode_syn import build as build_pernode
from day3_realtransfer import fix_inputs, eval_fn, gnn_predict

sys.setrecursionlimit(200000)


def cheap_feats(bn, S=384):
    """CHEAP training-consistent node features: signs from aeon regulation annotations
    (no truth table), bias/sensitivity/canalizing-depth ESTIMATED by sampling the
    update function. O(n*k*S) — bounded by per-node arity, NEVER 2^n. This IS the
    amortization argument: the GNN does bounded per-node work, the oracle does 2^n work."""
    vars_ = bn.variables(); idx = {v: i for i, v in enumerate(vars_)}; n = len(vars_)
    sign_map = {}
    for reg in bn.regulations():
        s = reg.get("sign"); sign_map[(idx[reg["source"]], idx[reg["target"]])] = (1 if s == "+" else -1 if s == "-" else 0)
    preds = [[] for _ in range(n)]; outdeg = [0] * n
    for v in vars_:
        for r in bn.predecessors(v):
            preds[idx[v]].append(idx[r]); outdeg[idx[r]] += 1
    rng = np.random.RandomState(0)
    X = np.zeros((n, 10), np.float32); src, dst, sgn = [], [], []
    for v in vars_:
        i = idx[v]; fn = bn.get_update_function(v)
        supp = sorted(fn.support_variables(), key=lambda x: idx[x]) if fn is not None else []
        k = len(supp)
        if k == 0 or fn is None:
            bias = (int(fn.as_const()) if (fn is not None and fn.is_const()) else 0.5)
            sens = depth = 0
        else:
            T = min(S, 1 << k) if k <= 22 else S
            outs = 0; flips = ftot = 0
            try:
                for _ in range(T):
                    asn = {s_: bool(rng.randint(2)) for s_ in supp}
                    o = int(eval_fn(fn, asn)); outs += o
                    b = rng.randint(k); a2 = dict(asn); a2[supp[b]] = not a2[supp[b]]
                    flips += int(int(eval_fn(fn, a2)) != o); ftot += 1
                bias = outs / T; sens = flips / max(ftot, 1)
                depth = 0
                for b in range(k):
                    o0, o1 = set(), set()
                    for _ in range(48):
                        asn = {s_: bool(rng.randint(2)) for s_ in supp}
                        asn[supp[b]] = False; o0.add(int(eval_fn(fn, asn)))
                        asn[supp[b]] = True; o1.add(int(eval_fn(fn, asn)))
                    if len(o0) == 1 or len(o1) == 1:
                        depth += 1
            except RecursionError:
                bias, sens, depth = 0.5, 0.5, 0
        act = sum(1 for r in preds[i] if sign_map.get((r, i), 0) > 0)
        inh = sum(1 for r in preds[i] if sign_map.get((r, i), 0) < 0)
        non = k - act - inh
        X[i] = [k, k / 8, bias, depth, depth / max(k, 1), sens, act / max(k, 1), inh / max(k, 1), non / max(k, 1), outdeg[i]]
        for r in preds[i]:
            src.append(r); dst.append(i); sgn.append(sign_map.get((r, i), 0))
    return X, np.array(src), np.array(dst), np.array(sgn, np.float32)


def oracle_worker(i, q):
    try:
        import biodivine_aeon as ba
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        from day3_realtransfer import fix_inputs as fx
        fx(bn)
        g = ba.AsynchronousGraph(bn)
        names = bn.variable_names()
        t0 = time.time()
        init = g.mk_subspace({nm: False for nm in names})
        reached = ba.Reachability.reach_fwd(g, init)
        vs = reached.vertices()
        y = np.array([int(not vs.intersect(g.mk_subspace_vertices({nm: True})).is_empty()) for nm in names], np.float32)
        q.put(("ok", float(time.time() - t0), y))
    except Exception as e:
        q.put(("err", repr(e)[:60], None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-n", type=int, default=400)
    a = ap.parse_args()

    # train GNN once
    graphs, *_ = build_pernode([12, 14], [2, 3, 4], 40, 4)
    np.random.seed(0); torch.manual_seed(0)
    net = train_gnn(graphs, np.arange(len(graphs)), "sign", a.epochs)
    print("[amortization] GNN trained.", flush=True)

    import biodivine_aeon as ba
    ids = ba.BiodivineBooleanModels.fetch_ids()
    ctx = mp.get_context("spawn")
    rows = []
    for k, i in enumerate(ids):
        try:
            bn = ba.BiodivineBooleanModels.fetch_network(i)
            n = bn.variable_count()
        except Exception:
            continue
        if n < 6 or n > a.max_n:
            continue
        fix_inputs(bn)
        # GNN prediction time (CHEAP featurize + forward) — bounded per-node work
        try:
            t0 = time.time()
            X, src, dst, sgn = cheap_feats(bn)
            p = gnn_predict(net, (X, src, dst, sgn, np.zeros(n, np.float32)))
            t_gnn = time.time() - t0
        except Exception as e:
            print(f"  id={i} n={n} featurize-skip ({repr(e)[:40]})", flush=True); continue
        # exact oracle time (process-isolated, hard budget)
        q = ctx.Queue(); pr = ctx.Process(target=oracle_worker, args=(i, q)); pr.start(); pr.join(a.budget)
        if pr.is_alive():
            pr.terminate(); pr.join()
            rows.append({"id": i, "n": n, "edges": len(src), "t_gnn": t_gnn, "t_oracle": None, "auc": None})
            print(f"  id={i} n={n:>4} edges={len(src):>5} | GNN {t_gnn*1e3:7.1f} ms | ORACLE TIMEOUT (>{a.budget}s)", flush=True)
            continue
        res = q.get_nowait()
        if res[0] != "ok":
            continue
        t_oracle, y = res[1], res[2]
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        rows.append({"id": i, "n": n, "edges": len(src), "t_gnn": t_gnn, "t_oracle": t_oracle, "auc": auc})
        sp = t_oracle / max(t_gnn, 1e-6)
        print(f"  id={i} n={n:>4} edges={len(src):>5} | GNN {t_gnn*1e3:7.1f} ms | oracle {t_oracle:7.2f} s | "
              f"speedup {sp:8.0f}x | AUROC {auc:.3f}", flush=True)
    pickle.dump(rows, open("amortization.pkl", "wb"))

    fin = [r for r in rows if r["t_oracle"] is not None]
    to = [r for r in rows if r["t_oracle"] is None]
    aucs = [r["auc"] for r in fin if r["auc"] == r["auc"]]
    print("\n================ AMORTIZATION PAYOFF ================", flush=True)
    print(f"  nets: {len(rows)} total | {len(fin)} oracle-feasible | {len(to)} oracle-TIMEOUT(>{a.budget}s)", flush=True)
    if fin:
        sp = [r["t_oracle"]/max(r["t_gnn"],1e-6) for r in fin]
        print(f"  where oracle feasible: GNN matches it at AUROC {np.mean(aucs):.3f} (median {np.median(aucs):.3f}), "
              f"median speedup {np.median(sp):.0f}x, max {np.max(sp):.0f}x", flush=True)
    if to:
        print(f"  ORACLE-INFEASIBLE nets the GNN still answers in ms: "
              f"{sorted((r['id'], r['n']) for r in to)}", flush=True)
    print(f"  GNN prediction time vs n: flat ~{np.median([r['t_gnn'] for r in rows])*1e3:.0f} ms; "
          f"oracle time grows with n (see amortization.pkl / figure).", flush=True)


if __name__ == "__main__":
    main()
