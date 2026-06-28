import biodivine_aeon as ba
import time

ids = ba.BiodivineBooleanModels.fetch_ids()[:12]
info = []
for i in ids:
    try:
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        info.append((i, bn.variable_count()))
    except Exception as e:
        print("fetch fail", i, repr(e)[:80])
info.sort(key=lambda x: x[1])
print("sizes (id, n):", info)

for i, n in info[:3]:
    bn = ba.BiodivineBooleanModels.fetch_network(i)
    g = ba.AsynchronousGraph(bn)
    t0 = time.time()
    atts = ba.Attractors.attractors(g)
    n_att = len(atts)
    # complex vs fixed: a fixed-point attractor has exactly 1 vertex
    n_complex = sum(1 for a in atts if a.vertices().cardinality() > 1)
    t1 = time.time()
    # symbolic forward reachability from the all-false corner
    init = g.mk_subspace_vertices({v: False for v in bn.variable_names()})
    reached = ba.Reachability.reach_fwd(g, init)
    t2 = time.time()
    print(f"id={i} n={n}: #attractors={n_att} #complex(cyclic)={n_complex} "
          f"[attr {t1-t0:.1f}s] | fwd-reach size={reached.cardinality():.3g} [reach {t2-t1:.1f}s]")
print("BBM TOOLCHAIN OK")
