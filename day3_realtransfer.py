"""Day-3b CAPSTONE — the GNN-necessity proof: train the per-node activation-reachability
predictor on SYNTHETIC small Boolean nets, then ZERO-SHOT TRANSFER to REAL biodivine-
boolean-models GRNs (n up to ~64) where the small-n flattened-truth-table GBDT is
INAPPLICABLE. Compare the size-agnostic GNN vs the size-agnostic per-node-feature GBDT.

Each real net is converted to the SAME (inputs, truth-table) representation used for
synthetic nets (evaluate each node's aeon update function over its support variables),
so features/graph/labels are computed identically. Real per-node labels come from
aeon's symbolic reach_fwd(all-false) ∩ {v=1}. GO = GNN transfers and beats the GBDT
on real nets (its propagation bias is the size-agnostic, scalable learner).
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import biodivine_aeon as ba
from decisive_gate import make_logic
from gnn_sparse import to_sparse, arity_inv_node_feats, DEV
from day3_pernode_syn import PerNodeGNN, pernode_activation, collate
from attractor_gonogo import sync_attractors  # noqa


# --------------------------------------------------------------------------- #
# real net -> (inputs, truth-table) per node by evaluating aeon update functions
# --------------------------------------------------------------------------- #
def eval_fn(fn, assign):
    if fn.is_const():
        return bool(fn.as_const())
    if fn.is_var():
        return assign[fn.as_var()]
    if fn.is_not():
        return not eval_fn(fn.as_not(), assign)
    if fn.is_and():
        a, b = fn.as_and(); return eval_fn(a, assign) and eval_fn(b, assign)
    if fn.is_or():
        a, b = fn.as_or(); return eval_fn(a, assign) or eval_fn(b, assign)
    if fn.is_imp():
        a, b = fn.as_imp(); return (not eval_fn(a, assign)) or eval_fn(b, assign)
    if fn.is_iff():
        a, b = fn.as_iff(); return eval_fn(a, assign) == eval_fn(b, assign)
    if fn.is_xor():
        a, b = fn.as_xor(); return eval_fn(a, assign) != eval_fn(b, assign)
    raise ValueError("param/unsupported node (parameterized function)")


def fix_inputs(bn):
    """Set every INPUT node (no update function AND no regulators = free external signal)
    to constant FALSE, so the net is fully specified and reach_fwd from the all-OFF state
    is deterministic. Returns #inputs fixed. (Narrative: full quiescence, no external signals.)"""
    cnt = 0
    for v in bn.variables():
        if bn.get_update_function(v) is None and len(bn.predecessors(v)) == 0:
            bn.set_update_function(v, "false"); cnt += 1
    return cnt


def real_net_to_repr(bn, max_k=18):
    """Return (inputs, tt) per node, or None if a node is an implicit parameter or arity too high.
    Call fix_inputs(bn) FIRST so true input nodes are constants, not None."""
    vars_ = bn.variables()
    idx = {v: i for i, v in enumerate(vars_)}
    n = len(vars_)
    inputs, tt = [], []
    for v in vars_:
        fn = bn.get_update_function(v)
        if fn is None:
            return None
        supp = sorted(fn.support_variables(), key=lambda x: idx[x])
        k = len(supp)
        if k > max_k:
            return None
        try:
            table = []
            for x in range(1 << k):
                assign = {supp[b]: bool((x >> b) & 1) for b in range(k)}
                table.append(int(eval_fn(fn, assign)))
        except ValueError:
            return None
        inputs.append([idx[s] for s in supp]); tt.append(table)
    return inputs, tt


def real_pernode_labels(bn):
    """Per-node activation reachability from all-false via aeon symbolic reach_fwd."""
    g = ba.AsynchronousGraph(bn)
    names = bn.variable_names()
    init = g.mk_subspace({nm: False for nm in names})
    reached = ba.Reachability.reach_fwd(g, init)
    vset = reached.vertices()
    on = []
    for nm in names:
        sub = g.mk_subspace_vertices({nm: True})
        inter = vset.intersect(sub)
        on.append(int(inter.cardinality() > 0))
    return np.array(on, np.float32)


def build_synth(n_list, K_list, gpc, v_each, seed=7):
    rng = random.Random(seed)
    graphs = []
    Xnode, ynode = [], []
    for n in n_list:
        for K in K_list:
            for _ in range(gpc):
                inputs = [rng.sample(range(n), min(K, n)) for _ in range(n)]
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        tt = make_logic(inputs, mode, rng)
                        X, src, dst, sgn = to_sparse(inputs, tt, n)
                        y = np.array(pernode_activation(inputs, tt, n), np.float32)
                        graphs.append((X, src, dst, sgn, y))
                        Xnode.append(X); ynode.append(y)
    return graphs, np.vstack(Xnode), np.concatenate(ynode)


def train_gnn(graphs, epochs, bs=128):
    net = PerNodeGNN(graphs[0][0].shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    idx_all = np.arange(len(graphs))
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx_all)
        for i in range(0, len(perm), bs):
            X, src, dst, sgn, y = collate(graphs, perm[i:i + bs], DEV)
            opt.zero_grad(); lossf(net(X, src, dst, sgn), y).backward(); opt.step()
    return net


def gnn_predict(net, graph):
    X, src, dst, sgn, y = graph
    cat = lambda a, dt: torch.tensor(a, dtype=dt, device=DEV)
    with torch.no_grad():
        p = net(cat(X, torch.float32), cat(src, torch.long), cat(dst, torch.long),
                cat(sgn, torch.float32)).float().cpu().numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpc", type=int, default=40)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--n-real", type=int, default=120)
    a = ap.parse_args()
    print(f"[day3 real-transfer] device={DEV}", flush=True)

    # 1) train on synthetic
    graphs, Xnode, ynode = build_synth([12, 14], [2, 3, 4], a.gpc, 4)
    print(f"  synthetic: {len(graphs)} nets, {len(ynode)} nodes, base {ynode.mean():.2f}", flush=True)
    net = train_gnn(graphs, a.epochs)
    gb = HistGradientBoostingClassifier(max_iter=400).fit(Xnode, ynode)
    print("  trained GNN + per-node GBDT on synthetic.", flush=True)

    # 2) build real-net test set
    ids = ba.BiodivineBooleanModels.fetch_ids()
    realX, realY, realP_gnn = [], [], []
    per_net = []
    used = skipped = 0
    for i in ids:
        if used >= a.n_real:
            break
        try:
            bn = ba.BiodivineBooleanModels.fetch_network(i)
            n = bn.variable_count()
            if n < 6 or n > a.max_n:
                continue
            fix_inputs(bn)                       # input nodes -> constant false (consistent repr + labels)
            rep = real_net_to_repr(bn)
            if rep is None:
                skipped += 1; continue
            inputs, tt = rep
            y = real_pernode_labels(bn)
            X, src, dst, sgn = to_sparse(inputs, tt, n)
            p = gnn_predict(net, (X, src, dst, sgn, y))
            realX.append(X); realY.append(y); realP_gnn.append(p)
            if len(np.unique(y)) > 1:
                per_net.append(roc_auc_score(y, p))
            used += 1
        except Exception as e:
            skipped += 1
            continue
    print(f"  real nets used={used} skipped(param/high-arity/err)={skipped}", flush=True)

    Yr = np.concatenate(realY); Pr = np.concatenate(realP_gnn); Xr = np.vstack(realX)
    gnn_auc = roc_auc_score(Yr, Pr) if len(np.unique(Yr)) > 1 else float("nan")
    gb_auc = roc_auc_score(Yr, gb.predict_proba(Xr)[:, 1]) if len(np.unique(Yr)) > 1 else float("nan")
    print(f"\n================ REAL-BBM TRANSFER (pooled over {len(Yr)} real nodes, base {Yr.mean():.2f}) ================", flush=True)
    print(f"  GNN (size-agnostic, synth-trained)  AUROC {gnn_auc:.3f}", flush=True)
    print(f"  per-node-feature GBDT (synth-trained) AUROC {gb_auc:.3f}", flush=True)
    print(f"  per-net GNN AUROC: median {np.median(per_net):.3f} over {len(per_net)} nets with both classes", flush=True)
    print(f"  GNN minus GBDT on REAL: {gnn_auc - gb_auc:+.3f}", flush=True)
    print("  GO = GNN transfers (AUROC well above 0.5) AND >= the GBDT; the flat-TT GBDT is INAPPLICABLE to these sizes -> GNN is the necessary scalable learner.", flush=True)


if __name__ == "__main__":
    main()
