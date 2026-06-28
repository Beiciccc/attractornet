import biodivine_aeon as ba
import time

ids = ba.BiodivineBooleanModels.fetch_ids()
nets = []
for i in ids[:40]:
    try:
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        nets.append((i, bn.variable_count(), bn))
    except Exception as e:
        pass
nets.sort(key=lambda x: x[1])
print("fetched", len(nets), "sizes:", [(i,n) for i,n,_ in nets])

for i, n, bn in nets:
    if n > 70:
        continue
    g = ba.AsynchronousGraph(bn)
    rec = {"id": i, "n": n}
    try:
        t0 = time.time()
        atts = ba.Attractors.attractors(g)
        rec["natt"] = len(atts)
        rec["ncomplex"] = sum(1 for a in atts if a.vertices().cardinality() > 1)
        rec["t_attr"] = round(time.time() - t0, 2)
    except Exception as e:
        rec["attr_err"] = repr(e)[:60]
    try:
        t1 = time.time()
        init = g.mk_subspace({v: False for v in bn.variable_names()})   # ColoredVertexSet
        reached = ba.Reachability.reach_fwd(g, init)
        rec["reach_card"] = float(reached.vertices().cardinality())
        rec["t_reach"] = round(time.time() - t1, 2)
    except Exception as e:
        rec["reach_err"] = repr(e)[:60]
    print(rec, flush=True)
print("DONE")
