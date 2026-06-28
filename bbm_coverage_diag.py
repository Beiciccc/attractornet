import biodivine_aeon as ba
from collections import Counter
import numpy as np

ids = ba.BiodivineBooleanModels.fetch_ids()
cat = Counter()
maxk_hist = []
usable_by_maxk = {12:0, 16:0, 20:0, 9999:0}
n_ok_fullyspec = 0
sizes_ok = []
for i in ids:
    try:
        bn = ba.BiodivineBooleanModels.fetch_network(i)
        n = bn.variable_count()
        # parameters?
        has_param = (bn.explicit_parameter_count() > 0) or (bn.implicit_parameter_count() > 0)
        # max in-degree (support size) across nodes; also detect None update fns
        maxk = 0; none_fn = False
        for v in bn.variables():
            fn = bn.get_update_function(v)
            if fn is None:
                none_fn = True; break
            maxk = max(maxk, len(fn.support_variables()))
        if none_fn:
            cat["none_update_fn"] += 1; continue
        if has_param:
            cat["parameterized"] += 1; continue
        cat["fully_specified"] += 1
        n_ok_fullyspec += 1
        maxk_hist.append(maxk)
        sizes_ok.append(n)
        for cap in usable_by_maxk:
            if maxk <= cap:
                usable_by_maxk[cap] += 1
    except Exception as e:
        cat["fetch/other_err"] += 1

print("category counts:", dict(cat))
print("fully-specified nets:", n_ok_fullyspec)
print("among fully-specified, usable by max-in-degree cap:", usable_by_maxk)
if maxk_hist:
    mh = np.array(maxk_hist)
    print(f"max-in-degree among fully-spec: min{mh.min()} median{int(np.median(mh))} p90{int(np.percentile(mh,90))} max{mh.max()}")
    sz = np.array(sizes_ok)
    print(f"sizes(n) fully-spec: min{sz.min()} median{int(np.median(sz))} p90{int(np.percentile(sz,90))} max{sz.max()}")
    print(f"fully-spec nets with n<=64: {int((sz<=64).sum())}; n<=100: {int((sz<=100).sum())}")
