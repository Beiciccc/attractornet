import biodivine_aeon as ba
# find a net with input nodes
ids = ba.BiodivineBooleanModels.fetch_ids()
for i in ids:
    bn = ba.BiodivineBooleanModels.fetch_network(i)
    nones = [v for v in bn.variables() if bn.get_update_function(v) is None]
    if nones:
        v = nones[0]
        print(f"net {i} n={bn.variable_count()} #None={len(nones)}")
        print("  is_variable_input:", bn.is_variable_input(v), "predecessors:", len(bn.predecessors(v)))
        # how many None nodes have 0 regulators (true inputs) vs >0 (implicit params)
        inp = sum(1 for x in nones if len(bn.predecessors(x))==0)
        print(f"  None nodes: {len(nones)} total, {inp} are true inputs (0 regulators), {len(nones)-inp} have regulators")
        # try setting update function to false
        for arg in ["false", "0"]:
            try:
                bn.set_update_function(v, arg); print(f"  set_update_function(v,'{arg}') OK -> now None? {bn.get_update_function(v) is None}"); break
            except Exception as e:
                print(f"  set('{arg}') err:", repr(e)[:70])
        break
