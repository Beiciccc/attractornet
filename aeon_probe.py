import biodivine_aeon as ba
bn = ba.BiodivineBooleanModels.fetch_network('3')   # n=20
print("vars:", bn.variable_count())
v0 = bn.variables()[5]
print("var name:", bn.get_variable_name(v0))
print("BooleanNetwork methods:", [m for m in dir(bn) if not m.startswith('_')])
print()
# regulators + signs
print("regulators(v0):", bn.regulators(v0))
fn = bn.get_update_function(v0)
print("update fn type:", type(fn).__name__, "->", str(fn)[:120])
print("UpdateFunction methods:", [m for m in dir(fn) if not m.startswith('_')])
