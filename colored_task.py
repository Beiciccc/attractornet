"""#1: FREE-INPUT COLORED REACHABILITY task (non-degenerate, matched synthetic<->real).

Networks now have FREE EXTERNAL INPUT nodes (in-degree 0, no update function = free
signal), like real GRNs. Task (per node v): is v=1 reachable for SOME input
configuration, starting from internal quiescence (all internal nodes OFF, inputs
free)? Biologically: is gene v activatable by some external signal. This removes the
'all-OFF is a quiescent fixed point' degeneracy of the inputs-OFF task and makes the
synthetic ensemble the same KIND of object as real BBM nets.

This module: synthetic generator with inputs + EXACT colored-activation oracle +
is_input-augmented arity-invariant features + an in-distribution GNN-vs-GBDT check.
"""
import argparse
import random
from collections import deque
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from attractor_gonogo import node_next, edge_sign
from gnn_sparse import DEV
from day3_pernode_syn import PerNodeGNN, collate


# --------------------------------------------------------------------------- #
# synthetic nets WITH free input nodes                                         #
# --------------------------------------------------------------------------- #
def gen_bn_inputs(n, n_inputs, K, mode, rng):
    """input nodes: in-degree 0, tt=None (free). internal: K regulators from ALL nodes."""
    input_ids = set(rng.sample(range(n), n_inputs))
    inputs, tt = [], []
    for i in range(n):
        if i in input_ids:
            inputs.append([]); tt.append(None); continue
        k = min(K, n)
        ins = rng.sample(range(n), k)
        if mode == "canalizing":
            from attractor_gonogo import nested_canalizing_tt
            table = nested_canalizing_tt(k, rng)
        else:
            p = rng.uniform(0.3, 0.7)
            table = [1 if rng.random() < p else 0 for _ in range(1 << k)]
        inputs.append(ins); tt.append(table)
    mask = np.array([1 if i in input_ids else 0 for i in range(n)], np.int8)
    return inputs, tt, mask


def colored_activation(inputs, tt, mask, n):
    """EXACT: per node v, label=1 if v=1 reachable from internal-OFF for SOME input config.
    Inputs frozen per config; async BFS flips only internal nodes."""
    inp_idx = [i for i in range(n) if mask[i]]
    internal = [i for i in range(n) if not mask[i]]
    m = len(inp_idx)
    reached_or = 0
    for c in range(1 << m):
        s0 = 0
        for b, j in enumerate(inp_idx):
            if (c >> b) & 1:
                s0 |= (1 << j)                     # set input j to its config bit
        seen = {s0}; dq = deque([s0]); acc = s0
        while dq:
            s = dq.popleft()
            for i in internal:
                if node_next(s, i, inputs, tt) != ((s >> i) & 1):
                    ns = s ^ (1 << i)
                    if ns not in seen:
                        seen.add(ns); dq.append(ns); acc |= ns
        reached_or |= acc
    return [(reached_or >> v) & 1 for v in range(n)]


# --------------------------------------------------------------------------- #
# is_input-augmented arity-invariant node features (11 dims)                    #
# --------------------------------------------------------------------------- #
def node_feats(inputs, tt, mask, n):
    outdeg = [0] * n
    for i in range(n):
        for j in inputs[i]:
            outdeg[j] += 1
    F = np.zeros((n, 11), np.float32)
    for i in range(n):
        if mask[i]:                               # free input node
            F[i] = [0, 0, 0.5, 0, 0, 0, 0, 0, 0, outdeg[i], 1.0]
            continue
        table = tt[i]; k = len(inputs[i])
        if (1 << k) <= 4096:
            sflip = sum(1 for x in range(1 << k) for b in range(k) if table[x] != table[x ^ (1 << b)])
            sens = sflip / max((1 << k) * k, 1)
        else:
            rng = random.Random(i); s = t = 0
            for _ in range(4096):
                x = rng.randrange(1 << k); b = rng.randrange(k)
                s += int(table[x] != table[x ^ (1 << b)]); t += 1
            sens = s / max(t, 1)
        depth = act = inh = non = 0
        for b in range(k):
            v0 = {table[x] for x in range(len(table)) if not ((x >> b) & 1)}
            v1 = {table[x] for x in range(len(table)) if ((x >> b) & 1)}
            if len(v0) == 1 or len(v1) == 1:
                depth += 1
            sg = edge_sign(inputs, tt, i, b)
            act += sg > 0; inh += sg < 0; non += sg == 0
        F[i] = [k, k / 8, sum(table) / len(table), depth, depth / max(k, 1), sens,
                act / max(k, 1), inh / max(k, 1), non / max(k, 1), outdeg[i], 0.0]
    return F


def to_sparse_ci(inputs, tt, mask, n):
    X = node_feats(inputs, tt, mask, n)
    src, dst, sgn = [], [], []
    for i in range(n):
        if mask[i]:
            continue
        for b, j in enumerate(inputs[i]):
            src.append(j); dst.append(i); sgn.append(edge_sign(inputs, tt, i, b))
    return X, np.array(src), np.array(dst), np.array(sgn, np.float32)


def build_synth(n_list, K_list, gpc, v_each, seed=7):
    rng = random.Random(seed)
    graphs, Xnode, ynode = [], [], []
    for n in n_list:
        for K in K_list:
            for _ in range(gpc):
                n_inputs = rng.randint(1, max(1, n // 5))
                base_inputs = None
                for mode in ("random", "canalizing"):
                    for _ in range(v_each):
                        inputs, tt, mask = gen_bn_inputs(n, n_inputs, K, mode, rng)
                        y = np.array(colored_activation(inputs, tt, mask, n), np.float32)
                        X, src, dst, sgn = to_sparse_ci(inputs, tt, mask, n)
                        graphs.append((X, src, dst, sgn, y))
                        Xnode.append(X); ynode.append(y)
    return graphs, np.vstack(Xnode), np.concatenate(ynode)


def train_gnn(graphs, epochs, bs=128):
    import torch.nn as nn
    net = PerNodeGNN(graphs[0][0].shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    idx = np.arange(len(graphs))
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            X, s, d, g, y = collate(graphs, perm[i:i + bs], DEV)
            opt.zero_grad(); lossf(net(X, s, d, g), y).backward(); opt.step()
    return net


def gnn_predict(net, graph):
    X, src, dst, sgn, y = graph
    cat = lambda a, dt: torch.tensor(a, dtype=dt, device=DEV)
    with torch.no_grad():
        return net(cat(X, torch.float32), cat(src, torch.long), cat(dst, torch.long), cat(sgn, torch.float32)).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--gpc", type=int, default=40); a = ap.parse_args()
    print(f"[colored-task synthetic in-dist] device={DEV}", flush=True)
    graphs, Xnode, ynode = build_synth([12, 14], [2, 3, 4], a.gpc, 4)
    print(f"  {len(graphs)} nets, {len(ynode)} nodes, base {ynode.mean():.2f}", flush=True)
    rs = np.random.RandomState(0); idx = rs.permutation(len(graphs))
    tr, te = idx[:int(.7*len(idx))], idx[int(.7*len(idx)):]
    starts = np.cumsum([0] + [g[0].shape[0] for g in graphs])
    rows = lambda S: np.concatenate([np.arange(starts[i], starts[i + 1]) for i in S])
    net = train_gnn([graphs[i] for i in tr], a.epochs)
    P = np.concatenate([gnn_predict(net, graphs[i]) for i in te]); Yte = np.concatenate([graphs[i][4] for i in te])
    gnn_auc = roc_auc_score(Yte, P)
    gb = HistGradientBoostingClassifier(max_iter=400).fit(Xnode[rows(tr)], ynode[rows(tr)])
    gb_auc = roc_auc_score(ynode[rows(te)], gb.predict_proba(Xnode[rows(te)])[:, 1])
    print(f"\n== colored activation reachability (base {ynode.mean():.2f}) ==", flush=True)
    print(f"  GNN {gnn_auc:.3f}  vs  per-node-feature GBDT {gb_auc:.3f}  (Δ {gnn_auc-gb_auc:+.3f})", flush=True)


if __name__ == "__main__":
    main()
